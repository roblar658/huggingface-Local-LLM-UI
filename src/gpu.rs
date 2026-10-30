use crate::inference::GgmlDType;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static PTX_IMAGE: &[u8] = include_bytes!("gguf_kernels.ptx");

#[derive(Debug, Clone, Copy, Default)]
pub struct GpuTelemetry {
    pub gpu_util_pct: u32,
    pub mem_util_pct: u32,
    pub vram_used_mb: u64,
    pub vram_free_mb: u64,
    pub vram_total_mb: u64,
    pub temp_c: u32,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const u8) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const u8) -> *mut c_void;
}

// ============================================================================
// 1. SANNTIDS GPU-TELEMETRI VIA NVIDIA NVML (nvml.dll)
// ============================================================================

#[repr(C)]
struct NvmlUtilization {
    gpu: u32,
    memory: u32,
}

#[repr(C)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

type FnNvmlInit = unsafe extern "C" fn() -> i32;
type FnNvmlGetHandle = unsafe extern "C" fn(u32, *mut usize) -> i32;
type FnNvmlGetUtil = unsafe extern "C" fn(usize, *mut NvmlUtilization) -> i32;
type FnNvmlGetMem = unsafe extern "C" fn(usize, *mut NvmlMemory) -> i32;
type FnNvmlGetTemp = unsafe extern "C" fn(usize, u32, *mut u32) -> i32;

struct NvmlMonitor {
    dev: usize,
    get_util: FnNvmlGetUtil,
    get_mem: FnNvmlGetMem,
    get_temp: Option<FnNvmlGetTemp>,
}

static NVML_INSTANCE: OnceLock<Option<NvmlMonitor>> = OnceLock::new();

impl NvmlMonitor {
    fn init() -> Option<Self> {
        #[cfg(windows)]
        unsafe {
            let lib = LoadLibraryA(b"nvml.dll\0".as_ptr());
            if lib.is_null() {
                return None;
            }
            let init_ptr = GetProcAddress(lib, b"nvmlInit_v2\0".as_ptr());
            let handle_ptr = GetProcAddress(lib, b"nvmlDeviceGetHandleByIndex_v2\0".as_ptr());
            let util_ptr = GetProcAddress(lib, b"nvmlDeviceGetUtilizationRates\0".as_ptr());
            let mem_ptr = GetProcAddress(lib, b"nvmlDeviceGetMemoryInfo\0".as_ptr());
            let temp_ptr = GetProcAddress(lib, b"nvmlDeviceGetTemperature\0".as_ptr());

            if init_ptr.is_null()
                || handle_ptr.is_null()
                || util_ptr.is_null()
                || mem_ptr.is_null()
            {
                return None;
            }

            let nvml_init: FnNvmlInit = std::mem::transmute(init_ptr);
            let nvml_get_handle: FnNvmlGetHandle = std::mem::transmute(handle_ptr);
            let get_util: FnNvmlGetUtil = std::mem::transmute(util_ptr);
            let get_mem: FnNvmlGetMem = std::mem::transmute(mem_ptr);
            let get_temp: Option<FnNvmlGetTemp> = if !temp_ptr.is_null() {
                Some(std::mem::transmute(temp_ptr))
            } else {
                None
            };

            if nvml_init() != 0 {
                return None;
            }
            let mut dev = 0usize;
            if nvml_get_handle(0, &mut dev) != 0 {
                return None;
            }

            Some(Self {
                dev,
                get_util,
                get_mem,
                get_temp,
            })
        }
        #[cfg(not(windows))]
        {
            None
        }
    }
}

pub fn query_gpu_telemetry() -> Option<GpuTelemetry> {
    let mon = NVML_INSTANCE.get_or_init(NvmlMonitor::init).as_ref()?;
    unsafe {
        let mut util = NvmlUtilization { gpu: 0, memory: 0 };
        let _ = (mon.get_util)(mon.dev, &mut util);

        let mut mem = NvmlMemory {
            total: 0,
            free: 0,
            used: 0,
        };
        if (mon.get_mem)(mon.dev, &mut mem) != 0 {
            return None;
        }

        let mut temp_c = 0u32;
        if let Some(get_temp) = mon.get_temp {
            let _ = get_temp(mon.dev, 0, &mut temp_c);
        }

        Some(GpuTelemetry {
            gpu_util_pct: util.gpu,
            mem_util_pct: util.memory,
            vram_used_mb: mem.used / (1024 * 1024),
            vram_free_mb: mem.free / (1024 * 1024),
            vram_total_mb: mem.total / (1024 * 1024),
            temp_c,
        })
    }
}

// ============================================================================
// 2. EGEN CUDA DRIVER API (nvcuda.dll) + PTX GPU-AKSELERATOR FOR GGUF
// ============================================================================

type FnCuInit = unsafe extern "system" fn(u32) -> i32;
type FnCuDeviceGet = unsafe extern "system" fn(*mut i32, i32) -> i32;
type FnCuDeviceGetName = unsafe extern "system" fn(*mut u8, i32, i32) -> i32;
type FnCuCtxCreate = unsafe extern "system" fn(*mut usize, u32, i32) -> i32;
type FnCuCtxSetCurrent = unsafe extern "system" fn(usize) -> i32;
type FnCuMemGetInfo = unsafe extern "system" fn(*mut usize, *mut usize) -> i32;
type FnCuMemAlloc = unsafe extern "system" fn(*mut u64, usize) -> i32;
type FnCuMemFree = unsafe extern "system" fn(u64) -> i32;
type FnCuMemcpyHtoD = unsafe extern "system" fn(u64, *const c_void, usize) -> i32;
type FnCuMemcpyDtoH = unsafe extern "system" fn(*mut c_void, u64, usize) -> i32;
type FnCuModuleLoadData = unsafe extern "system" fn(*mut usize, *const c_void) -> i32;
type FnCuModuleGetFunction = unsafe extern "system" fn(*mut usize, usize, *const u8) -> i32;
type FnCuLaunchKernel = unsafe extern "system" fn(
    usize,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    usize,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;

struct CudaWorkspace {
    d_x: u64,
    x_cap_bytes: usize,
    d_out: u64,
    out_cap_bytes: usize,
    d_inter1: u64,
    inter1_cap_bytes: usize,
    d_inter2: u64,
    inter2_cap_bytes: usize,
}

pub struct CudaEngine {
    pub device_name: String,
    ctx: usize,
    cu_ctx_set_current: FnCuCtxSetCurrent,
    cu_mem_get_info: FnCuMemGetInfo,
    cu_mem_alloc: FnCuMemAlloc,
    cu_mem_free: FnCuMemFree,
    cu_memcpy_htod: FnCuMemcpyHtoD,
    cu_memcpy_dtoh: FnCuMemcpyDtoH,
    cu_launch_kernel: FnCuLaunchKernel,
    fn_q4k: usize,
    fn_q6k: usize,
    fn_q5k: usize,
    fn_q8_0: usize,
    fn_q4_0: usize,
    fn_f16: usize,
    fn_bf16: usize,
    fn_f32: usize,
    fn_swiglu: usize,
    workspace: Mutex<CudaWorkspace>,
    pub offloaded_bytes: AtomicUsize,
    pub offloaded_tensors: AtomicUsize,
    /// Dynamisk tak på antall aktive CUDA thread-blocks per kjerne-lansering (styres av ResourceGovernor)
    pub max_grid_x: AtomicU32,
}

unsafe impl Send for CudaEngine {}
unsafe impl Sync for CudaEngine {}

pub struct CudaWeightBuffer {
    pub d_ptr: u64,
    pub byte_len: usize,
    pub dtype: GgmlDType,
    engine: Arc<CudaEngine>,
}

unsafe impl Send for CudaWeightBuffer {}
unsafe impl Sync for CudaWeightBuffer {}

impl Drop for CudaWeightBuffer {
    fn drop(&mut self) {
        if self.d_ptr != 0 {
            unsafe {
                let _ = (self.engine.cu_ctx_set_current)(self.engine.ctx);
                let _ = (self.engine.cu_mem_free)(self.d_ptr);
            }
            self.engine
                .offloaded_bytes
                .fetch_sub(self.byte_len, Ordering::Relaxed);
            self.engine
                .offloaded_tensors
                .fetch_sub(1, Ordering::Relaxed);
            self.d_ptr = 0;
        }
    }
}

static CUDA_GLOBAL: OnceLock<Option<Arc<CudaEngine>>> = OnceLock::new();

impl CudaEngine {
    pub fn global() -> Option<Arc<CudaEngine>> {
        CUDA_GLOBAL.get_or_init(Self::init_internal).clone()
    }

    fn init_internal() -> Option<Arc<Self>> {
        #[cfg(windows)]
        unsafe {
            let lib = LoadLibraryA(b"nvcuda.dll\0".as_ptr());
            if lib.is_null() {
                return None;
            }

            macro_rules! load_sym {
                ($name:literal, $ty:ty) => {{
                    let p = GetProcAddress(lib, concat!($name, "\0").as_ptr());
                    if p.is_null() {
                        return None;
                    }
                    std::mem::transmute::<*mut c_void, $ty>(p)
                }};
            }

            let cu_init = load_sym!("cuInit", FnCuInit);
            let cu_device_get = load_sym!("cuDeviceGet", FnCuDeviceGet);
            let cu_device_get_name = load_sym!("cuDeviceGetName", FnCuDeviceGetName);
            let cu_ctx_create = load_sym!("cuCtxCreate_v2", FnCuCtxCreate);
            let cu_ctx_set_current = load_sym!("cuCtxSetCurrent", FnCuCtxSetCurrent);
            let cu_mem_get_info = load_sym!("cuMemGetInfo_v2", FnCuMemGetInfo);
            let cu_mem_alloc = load_sym!("cuMemAlloc_v2", FnCuMemAlloc);
            let cu_mem_free = load_sym!("cuMemFree_v2", FnCuMemFree);
            let cu_memcpy_htod = load_sym!("cuMemcpyHtoD_v2", FnCuMemcpyHtoD);
            let cu_memcpy_dtoh = load_sym!("cuMemcpyDtoH_v2", FnCuMemcpyDtoH);
            let cu_module_load_data = load_sym!("cuModuleLoadData", FnCuModuleLoadData);
            let cu_module_get_function = load_sym!("cuModuleGetFunction", FnCuModuleGetFunction);
            let cu_launch_kernel = load_sym!("cuLaunchKernel", FnCuLaunchKernel);

            if cu_init(0) != 0 {
                return None;
            }
            let mut dev = 0i32;
            if cu_device_get(&mut dev, 0) != 0 {
                return None;
            }

            let mut name_buf = [0u8; 128];
            let _ = cu_device_get_name(name_buf.as_mut_ptr(), 128, dev);
            let name_end = name_buf
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(name_buf.len());
            let device_name = String::from_utf8_lossy(&name_buf[..name_end]).to_string();

            let mut ctx = 0usize;
            if cu_ctx_create(&mut ctx, 0, dev) != 0 {
                return None;
            }

            let mut module = 0usize;
            if cu_module_load_data(&mut module, PTX_IMAGE.as_ptr() as *const c_void) != 0 {
                return None;
            }

            let get_fn = |fname: &[u8]| -> Option<usize> {
                let mut f = 0usize;
                if cu_module_get_function(&mut f, module, fname.as_ptr()) == 0 {
                    Some(f)
                } else {
                    None
                }
            };

            let fn_q4k = get_fn(b"matmul_q4k\0")?;
            let fn_q6k = get_fn(b"matmul_q6k\0")?;
            let fn_q5k = get_fn(b"matmul_q5k\0")?;
            let fn_q8_0 = get_fn(b"matmul_q8_0\0")?;
            let fn_q4_0 = get_fn(b"matmul_q4_0\0")?;
            let fn_f16 = get_fn(b"matmul_f16\0")?;
            let fn_bf16 = get_fn(b"matmul_bf16\0")?;
            let fn_f32 = get_fn(b"matmul_f32\0")?;
            let fn_swiglu = get_fn(b"swiglu_f32\0")?;

            let x_cap_bytes = 16 * 1024 * 1024;
            let out_cap_bytes = 32 * 1024 * 1024;
            let inter1_cap_bytes = 16 * 1024 * 1024;
            let inter2_cap_bytes = 16 * 1024 * 1024;

            let mut d_x = 0u64;
            let mut d_out = 0u64;
            let mut d_inter1 = 0u64;
            let mut d_inter2 = 0u64;

            if cu_mem_alloc(&mut d_x, x_cap_bytes) != 0
                || cu_mem_alloc(&mut d_out, out_cap_bytes) != 0
                || cu_mem_alloc(&mut d_inter1, inter1_cap_bytes) != 0
                || cu_mem_alloc(&mut d_inter2, inter2_cap_bytes) != 0
            {
                return None;
            }

            println!(
                "CUDA Driver API initialisert på '{}' (9 PTX-kjerner lastet i VRAM)!",
                device_name
            );

            Some(Arc::new(Self {
                device_name,
                ctx,
                cu_ctx_set_current,
                cu_mem_get_info,
                cu_mem_alloc,
                cu_mem_free,
                cu_memcpy_htod,
                cu_memcpy_dtoh,
                cu_launch_kernel,
                fn_q4k,
                fn_q6k,
                fn_q5k,
                fn_q8_0,
                fn_q4_0,
                fn_f16,
                fn_bf16,
                fn_f32,
                fn_swiglu,
                workspace: Mutex::new(CudaWorkspace {
                    d_x,
                    x_cap_bytes,
                    d_out,
                    out_cap_bytes,
                    d_inter1,
                    inter1_cap_bytes,
                    d_inter2,
                    inter2_cap_bytes,
                }),
                offloaded_bytes: AtomicUsize::new(0),
                offloaded_tensors: AtomicUsize::new(0),
                max_grid_x: AtomicU32::new(144),
            }))
        }
        #[cfg(not(windows))]
        {
            None
        }
    }

    #[inline(always)]
    fn kernel_for_dtype(&self, dtype: GgmlDType) -> Option<usize> {
        match dtype {
            GgmlDType::Q4K => Some(self.fn_q4k),
            GgmlDType::Q6K => Some(self.fn_q6k),
            GgmlDType::Q5K => Some(self.fn_q5k),
            GgmlDType::Q8_0 => Some(self.fn_q8_0),
            GgmlDType::Q4_0 => Some(self.fn_q4_0),
            GgmlDType::F16 => Some(self.fn_f16),
            GgmlDType::BF16 => Some(self.fn_bf16),
            GgmlDType::F32 => Some(self.fn_f32),
            _ => None,
        }
    }

    /// Sjekker om det er trygt å allokere `bytes` i GPU VRAM uten å overskride sikkerhetsmarginen.
    pub fn can_safely_alloc(&self, bytes: usize) -> bool {
        if bytes == 0 {
            return true;
        }
        unsafe {
            if (self.cu_ctx_set_current)(self.ctx) != 0 {
                return false;
            }
            let mut free_vram = 0usize;
            let mut total_vram = 0usize;
            if (self.cu_mem_get_info)(&mut free_vram, &mut total_vram) != 0 {
                return false;
            }
            if let Some(tel) = query_gpu_telemetry() {
                if tel.vram_total_mb > 0 {
                    let nvml_free = (tel.vram_free_mb as usize) * 1024 * 1024;
                    let nvml_total = (tel.vram_total_mb as usize) * 1024 * 1024;
                    free_vram = free_vram.min(nvml_free);
                    total_vram = total_vram.max(nvml_total);
                }
            }

            let max_offload_bytes = (total_vram * 58) / 100;
            let current_offloaded = self.offloaded_bytes.load(Ordering::Relaxed);
            if current_offloaded + bytes > max_offload_bytes {
                return false;
            }

            let safety_reserve = (total_vram / 5).max(1600 * 1024 * 1024);
            if free_vram <= bytes + safety_reserve {
                return false;
            }
            true
        }
    }

    /// Returnerer true dersom GPU VRAM eller systemet er under merkbart minnepress
    pub fn is_vram_under_pressure(&self) -> bool {
        unsafe {
            if (self.cu_ctx_set_current)(self.ctx) != 0 {
                return true;
            }
            let mut free_vram = 0usize;
            let mut total_vram = 0usize;
            if (self.cu_mem_get_info)(&mut free_vram, &mut total_vram) != 0 {
                return true;
            }
            if let Some(tel) = query_gpu_telemetry() {
                if tel.vram_total_mb > 0 {
                    let nvml_free = (tel.vram_free_mb as usize) * 1024 * 1024;
                    let nvml_total = (tel.vram_total_mb as usize) * 1024 * 1024;
                    free_vram = free_vram.min(nvml_free);
                    total_vram = total_vram.max(nvml_total);
                }
            }

            let safety_reserve = (total_vram / 5).max(1600 * 1024 * 1024);
            let max_offload_bytes = (total_vram * 58) / 100;
            let current_offloaded = self.offloaded_bytes.load(Ordering::Relaxed);
            free_vram < safety_reserve || current_offloaded > max_offload_bytes
        }
    }

    /// Laster opp en GGUF-vekttensor til GPU VRAM kun dersom det er trygt over sikkerhetsmarginen
    /// (beholder alltid minst ~1.25 GB eller 16% av total VRAM ledig for OS, skjerm og andre programmer).
    pub fn try_upload_weight(
        self: &Arc<Self>,
        bytes: &[u8],
        dtype: GgmlDType,
    ) -> Option<Arc<CudaWeightBuffer>> {
        self.kernel_for_dtype(dtype)?;
        if bytes.is_empty() {
            return None;
        }

        if !self.can_safely_alloc(bytes.len()) {
            return None;
        }

        unsafe {
            if (self.cu_ctx_set_current)(self.ctx) != 0 {
                return None;
            }

            let mut d_ptr = 0u64;
            if (self.cu_mem_alloc)(&mut d_ptr, bytes.len()) != 0 || d_ptr == 0 {
                return None;
            }

            if (self.cu_memcpy_htod)(d_ptr, bytes.as_ptr() as *const c_void, bytes.len()) != 0 {
                let _ = (self.cu_mem_free)(d_ptr);
                return None;
            }

            self.offloaded_bytes.fetch_add(bytes.len(), Ordering::Relaxed);
            self.offloaded_tensors.fetch_add(1, Ordering::Relaxed);

            Some(Arc::new(CudaWeightBuffer {
                d_ptr,
                byte_len: bytes.len(),
                dtype,
                engine: self.clone(),
            }))
        }
    }

    #[inline(always)]
    unsafe fn launch_matmul_raw(
        &self,
        buf: &CudaWeightBuffer,
        d_x: u64,
        d_out: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> bool {
        let Some(func) = self.kernel_for_dtype(buf.dtype) else {
            return false;
        };
        let mut arg_w = buf.d_ptr;
        let mut arg_x = d_x;
        let mut arg_out = d_out;
        let mut arg_m = m as i32;
        let mut arg_n = n as i32;
        let mut arg_k = k as i32;

        let mut params: [*mut c_void; 6] = [
            &mut arg_w as *mut u64 as *mut c_void,
            &mut arg_x as *mut u64 as *mut c_void,
            &mut arg_out as *mut u64 as *mut c_void,
            &mut arg_m as *mut i32 as *mut c_void,
            &mut arg_n as *mut i32 as *mut c_void,
            &mut arg_k as *mut i32 as *mut c_void,
        ];

        let needed_blocks = ((n as u32) + 7) / 8;
        let cap = self.max_grid_x.load(Ordering::Relaxed).max(4);
        let grid_x = needed_blocks.min(cap).max(1);

        (self.cu_launch_kernel)(
            func,
            grid_x,
            m as u32,
            1,
            256,
            1,
            1,
            0,
            0,
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        ) == 0
    }

    pub fn matmul_batched(
        &self,
        buf: &CudaWeightBuffer,
        x: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        let x_bytes = m * k * 4;
        let out_bytes = m * n * 4;

        let ws = self.workspace.lock().ok()?;
        if x_bytes > ws.x_cap_bytes || out_bytes > ws.out_cap_bytes {
            return None;
        }

        unsafe {
            if (self.cu_ctx_set_current)(self.ctx) != 0 {
                return None;
            }
            if (self.cu_memcpy_htod)(ws.d_x, x.as_ptr() as *const c_void, x_bytes) != 0 {
                return None;
            }
            if !self.launch_matmul_raw(buf, ws.d_x, ws.d_out, m, n, k) {
                return None;
            }
            let mut out = vec![0.0f32; m * n];
            if (self.cu_memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, ws.d_out, out_bytes) != 0 {
                return None;
            }
            Some(out)
        }
    }

    /// Kjører flere projeksjoner som deler nøyaktig samme input `x` (f.eks. Q, K, V eller SSM QKV, Gate, Alpha, Beta)
    /// med kun ÉN PCIe-opplasting av `x` og back-to-back kjerne-lanseringer på GPU.
    pub fn matmul_group_same_input(
        &self,
        targets: &[(&CudaWeightBuffer, usize)],
        x: &[f32],
        m: usize,
        k: usize,
    ) -> Option<Vec<Vec<f32>>> {
        let x_bytes = m * k * 4;
        let total_n: usize = targets.iter().map(|&(_, n)| n).sum();
        let total_out_bytes = m * total_n * 4;

        let ws = self.workspace.lock().ok()?;
        if x_bytes > ws.x_cap_bytes || total_out_bytes > ws.out_cap_bytes {
            return None;
        }

        unsafe {
            if (self.cu_ctx_set_current)(self.ctx) != 0 {
                return None;
            }
            if (self.cu_memcpy_htod)(ws.d_x, x.as_ptr() as *const c_void, x_bytes) != 0 {
                return None;
            }

            let mut results = Vec::with_capacity(targets.len());
            let mut offset_bytes = 0usize;

            for &(buf, n) in targets {
                let d_out_slice = ws.d_out + (offset_bytes as u64);
                let slice_bytes = m * n * 4;
                if !self.launch_matmul_raw(buf, ws.d_x, d_out_slice, m, n, k) {
                    return None;
                }
                offset_bytes += slice_bytes;
            }

            offset_bytes = 0;
            for &(_, n) in targets {
                let d_out_slice = ws.d_out + (offset_bytes as u64);
                let slice_bytes = m * n * 4;
                let mut out_vec = vec![0.0f32; m * n];
                if (self.cu_memcpy_dtoh)(
                    out_vec.as_mut_ptr() as *mut c_void,
                    d_out_slice,
                    slice_bytes,
                ) != 0
                {
                    return None;
                }
                results.push(out_vec);
                offset_bytes += slice_bytes;
            }

            Some(results)
        }
    }

    /// Fullstendig fusjonert SwiGLU Feed-Forward Network på GPU:
    /// 1) `d_inter1 = ffn_in * w_gate^T`
    /// 2) `d_inter2 = ffn_in * w_up^T`
    /// 3) `d_inter1 = silu(d_inter1) * d_inter2` (i VRAM/L2-cache uten PCIe-kopiering)
    /// 4) `d_out = d_inter1 * w_down^T`
    pub fn ffn_swiglu_fused(
        &self,
        gate_buf: &CudaWeightBuffer,
        up_buf: &CudaWeightBuffer,
        down_buf: &CudaWeightBuffer,
        ffn_in: &[f32],
        m: usize,
        n_embd: usize,
        ffn_inter: usize,
    ) -> Option<Vec<f32>> {
        let in_bytes = m * n_embd * 4;
        let inter_bytes = m * ffn_inter * 4;
        let out_bytes = m * n_embd * 4;

        let ws = self.workspace.lock().ok()?;
        if in_bytes > ws.x_cap_bytes
            || inter_bytes > ws.inter1_cap_bytes
            || inter_bytes > ws.inter2_cap_bytes
            || out_bytes > ws.out_cap_bytes
        {
            return None;
        }

        unsafe {
            if (self.cu_ctx_set_current)(self.ctx) != 0 {
                return None;
            }
            if (self.cu_memcpy_htod)(ws.d_x, ffn_in.as_ptr() as *const c_void, in_bytes) != 0 {
                return None;
            }

            if !self.launch_matmul_raw(gate_buf, ws.d_x, ws.d_inter1, m, ffn_inter, n_embd) {
                return None;
            }
            if !self.launch_matmul_raw(up_buf, ws.d_x, ws.d_inter2, m, ffn_inter, n_embd) {
                return None;
            }

            let mut arg_gate = ws.d_inter1;
            let mut arg_up = ws.d_inter2;
            let mut arg_elems = (m * ffn_inter) as i32;
            let mut swiglu_params: [*mut c_void; 3] = [
                &mut arg_gate as *mut u64 as *mut c_void,
                &mut arg_up as *mut u64 as *mut c_void,
                &mut arg_elems as *mut i32 as *mut c_void,
            ];
            let needed_blocks = (((m * ffn_inter) as u32) + 255) / 256;
            let cap = self.max_grid_x.load(Ordering::Relaxed).max(4);
            let grid_x = needed_blocks.min(cap).max(1);

            if (self.cu_launch_kernel)(
                self.fn_swiglu,
                grid_x,
                1,
                1,
                256,
                1,
                1,
                0,
                0,
                swiglu_params.as_mut_ptr(),
                std::ptr::null_mut(),
            ) != 0
            {
                return None;
            }

            if !self.launch_matmul_raw(down_buf, ws.d_inter1, ws.d_out, m, n_embd, ffn_inter) {
                return None;
            }

            let mut out = vec![0.0f32; m * n_embd];
            if (self.cu_memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, ws.d_out, out_bytes) != 0 {
                return None;
            }
            Some(out)
        }
    }
}

