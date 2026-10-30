use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokenizers::Tokenizer;

// ============================================================================
// 1. EGEN WINDOWS ZERO-COPY MINNEKARTLEGGING (Win32 MapViewOfFile)
// ============================================================================

#[cfg(windows)]
mod mmap {
    use anyhow::{bail, Result};
    use std::fs::File;
    use std::os::windows::io::AsRawHandle;

    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x0004;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateFileMappingW(
            hFile: *mut core::ffi::c_void,
            lpFileMappingAttributes: *const core::ffi::c_void,
            flProtect: u32,
            dwMaximumSizeHigh: u32,
            dwMaximumSizeLow: u32,
            lpName: *const u16,
        ) -> *mut core::ffi::c_void;

        fn MapViewOfFile(
            hFileMappingObject: *mut core::ffi::c_void,
            dwDesiredAccess: u32,
            dwFileOffsetHigh: u32,
            dwFileOffsetLow: u32,
            dwNumberOfBytesToMap: usize,
        ) -> *const u8;

        fn UnmapViewOfFile(lpBaseAddress: *const u8) -> i32;
        fn CloseHandle(hObject: *mut core::ffi::c_void) -> i32;
    }

    pub struct MmapFile {
        ptr: *const u8,
        len: usize,
        mapping_handle: *mut core::ffi::c_void,
        _file: File,
    }

    unsafe impl Send for MmapFile {}
    unsafe impl Sync for MmapFile {}

    impl MmapFile {
        pub fn open(file: File) -> Result<Self> {
            let len = file.metadata()?.len() as usize;
            if len == 0 {
                bail!("Kan ikke minnekartlegge en tom GGUF-fil");
            }
            unsafe {
                let raw_handle = file.as_raw_handle() as *mut core::ffi::c_void;
                let mapping = CreateFileMappingW(
                    raw_handle,
                    std::ptr::null(),
                    PAGE_READONLY,
                    0,
                    0,
                    std::ptr::null(),
                );
                if mapping.is_null() {
                    bail!("Win32 CreateFileMappingW feilet ved åpning av GGUF-fil");
                }
                let ptr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
                if ptr.is_null() {
                    CloseHandle(mapping);
                    bail!("Win32 MapViewOfFile feilet ved minnekartlegging av GGUF-fil");
                }
                Ok(Self {
                    ptr,
                    len,
                    mapping_handle: mapping,
                    _file: file,
                })
            }
        }

        #[inline(always)]
        pub fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    impl Drop for MmapFile {
        fn drop(&mut self) {
            unsafe {
                if !self.ptr.is_null() {
                    UnmapViewOfFile(self.ptr);
                }
                if !self.mapping_handle.is_null() {
                    CloseHandle(self.mapping_handle);
                }
            }
        }
    }
}

use mmap::MmapFile;

// ============================================================================
// 2. EGEN GGUF v1/v2/v3 BINÆRPARSER (100% EGEN KODE)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    BF16,
}

impl GgmlDType {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            30 => Ok(Self::BF16),
            other => bail!("Ukjent GGML-datatype i GGUF: {}", other),
        }
    }

    #[inline(always)]
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    #[inline(always)]
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Self::U32(v) => Some(*v as usize),
            Self::I32(v) if *v >= 0 => Some(*v as usize),
            Self::U64(v) => Some(*v as usize),
            Self::I64(v) if *v >= 0 => Some(*v as usize),
            Self::U16(v) => Some(*v as usize),
            Self::U8(v) => Some(*v as usize),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            Self::U32(v) => Some(*v as f32),
            Self::I32(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensorDesc {
    /// Dimensjoner i standard rekkefølge `[outer.., inner]` (reversert fra GGMLs `[inner, outer..]`)
    pub dims: Vec<usize>,
    pub dtype: GgmlDType,
    /// Absolutt byte-offset i GGUF-filen (`data_offset + tensor_offset`)
    pub abs_offset: usize,
    pub byte_len: usize,
}

pub struct GgufModelFile {
    pub mmap: Arc<MmapFile>,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorDesc>,
}

struct ByteCursor<'a> {
    data: &'a [u8],
    pos: usize,
    version: u32,
}

impl<'a> ByteCursor<'a> {
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            bail!("Uventet slutt på GGUF-fil ved posisjon {}", self.pos);
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_bytes(1)?[0] as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_len(&mut self) -> Result<usize> {
        if self.version == 1 {
            Ok(self.read_u32()? as usize)
        } else {
            Ok(self.read_u64()? as usize)
        }
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_len()?;
        let bytes = self.read_bytes(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn read_value(&mut self, vtype: u32) -> Result<GgufValue> {
        match vtype {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                let elem_type = self.read_u32()?;
                let len = self.read_len()?;
                let mut arr = Vec::with_capacity(len.min(1024));
                for _ in 0..len {
                    arr.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            other => bail!("Ukjent GGUF metadata-verdi-type: {}", other),
        }
    }
}

impl GgufModelFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Kunne ikke åpne GGUF-fil: {:?}", path))?;
        let mmap = Arc::new(MmapFile::open(file)?);
        let data = mmap.as_slice();

        if data.len() < 16 || &data[0..4] != b"GGUF" {
            bail!("Filen er ikke en gyldig GGUF-fil (mangler 'GGUF'-header)");
        }
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if !(1..=3).contains(&version) {
            bail!("Ustøttet GGUF-versjon: {}", version);
        }

        let mut cur = ByteCursor {
            data,
            pos: 8,
            version,
        };

        let tensor_count = cur.read_len()?;
        let kv_count = cur.read_len()?;

        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = cur.read_string()?;
            let vtype = cur.read_u32()?;
            let val = cur.read_value(vtype)?;
            metadata.insert(key, val);
        }

        let mut raw_tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            let mut ggml_dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                ggml_dims.push(cur.read_len()?);
            }
            ggml_dims.reverse(); // Konverter fra [inner, outer..] til [outer.., inner]
            let dtype = GgmlDType::from_u32(cur.read_u32()?)?;
            let rel_offset = cur.read_u64()? as usize;
            raw_tensors.push((name, ggml_dims, dtype, rel_offset));
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_usize())
            .unwrap_or(32)
            .max(1);
        let pad = (alignment - (cur.pos % alignment)) % alignment;
        let data_offset = cur.pos + pad;

        let mut tensors = HashMap::with_capacity(tensor_count);
        for (name, dims, dtype, rel_offset) in raw_tensors {
            let elem_count: usize = dims.iter().product();
            let byte_len = (elem_count / dtype.block_size()) * dtype.type_size();
            let abs_offset = data_offset + rel_offset;
            if abs_offset + byte_len > data.len() {
                bail!(
                    "Tensor '{}' går utenfor GGUF-filens slutt ({} > {})",
                    name,
                    abs_offset + byte_len,
                    data.len()
                );
            }
            tensors.insert(
                name,
                GgufTensorDesc {
                    dims,
                    dtype,
                    abs_offset,
                    byte_len,
                },
            );
        }

        Ok(Self {
            mmap,
            metadata,
            tensors,
        })
    }

    pub fn get_usize(&self, key: &str) -> Result<usize> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_usize())
            .ok_or_else(|| anyhow!("Mangler heltalls-metadata '{}' i GGUF", key))
    }

    pub fn get_f32(&self, key: &str) -> Result<f32> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| anyhow!("Mangler flyttalls-metadata '{}' i GGUF", key))
    }

    pub fn tensor_no_gpu(&self, name: &str) -> Result<MmapWeight> {
        let desc = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Fant ikke tensor '{}' i GGUF", name))?;
        Ok(MmapWeight {
            mmap: self.mmap.clone(),
            dims: desc.dims.clone(),
            dtype: desc.dtype,
            abs_offset: desc.abs_offset,
            byte_len: desc.byte_len,
            gpu_buf: None,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<MmapWeight> {
        let desc = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Fant ikke tensor '{}' i GGUF", name))?;
        let slice = &self.mmap.as_slice()[desc.abs_offset..desc.abs_offset + desc.byte_len];
        let gpu_buf = crate::gpu::CudaEngine::global()
            .and_then(|cuda| cuda.try_upload_weight(slice, desc.dtype));
        Ok(MmapWeight {
            mmap: self.mmap.clone(),
            dims: desc.dims.clone(),
            dtype: desc.dtype,
            abs_offset: desc.abs_offset,
            byte_len: desc.byte_len,
            gpu_buf,
        })
    }

    pub fn tensor_f32_1d(&self, name: &str) -> Result<Vec<f32>> {
        let w = self.tensor_no_gpu(name)?;
        let elem_count: usize = w.dims.iter().product();
        let mut out = vec![0.0f32; elem_count];
        dequantize_slice(w.bytes(), w.dtype, &mut out)?;
        Ok(out)
    }

    pub fn tensor_f32_flat(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let w = self.tensor_no_gpu(name)?;
        let elem_count: usize = w.dims.iter().product();
        let mut out = vec![0.0f32; elem_count];
        dequantize_slice(w.bytes(), w.dtype, &mut out)?;
        Ok((out, w.dims))
    }
}

/// Zero-copy vektmatrise knyttet direkte til den minnekartlagte GGUF-filen (`MapViewOfFile`)
/// med valgfri direkte GPU VRAM-akselerasjon via CUDA PTX (`CudaWeightBuffer`).
#[derive(Clone)]
pub struct MmapWeight {
    mmap: Arc<MmapFile>,
    pub dims: Vec<usize>,
    pub dtype: GgmlDType,
    abs_offset: usize,
    byte_len: usize,
    pub gpu_buf: Option<Arc<crate::gpu::CudaWeightBuffer>>,
}

#[derive(Debug, Clone, Default)]
pub struct HybridStatus {
    pub total_layers: usize,
    pub gpu_layers: usize,
    pub cpu_layers: usize,
    pub gpu_vram_bytes: usize,
    pub output_on_gpu: bool,
}

impl HybridStatus {
    pub fn summary(&self) -> String {
        let vram_gb = (self.gpu_vram_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
        if self.gpu_layers == self.total_layers && self.total_layers > 0 {
            format!("Full GPU (alle {}/{} lag i VRAM: {:.2} GB)", self.gpu_layers, self.total_layers, vram_gb)
        } else if self.gpu_layers > 0 {
            format!(
                "Hybrid GPU+CPU (dynamisk): {}/{} lag i GPU ({:.2} GB VRAM) + {} lag på CPU",
                self.gpu_layers, self.total_layers, vram_gb, self.cpu_layers
            )
        } else {
            format!("Ren CPU (alle {} lag i systemminne via AVX2+FMA)", self.total_layers)
        }
    }
}

impl MmapWeight {
    pub fn try_promote_to_gpu(&mut self) -> bool {
        if self.gpu_buf.is_some() {
            return true;
        }
        let Some(cuda) = crate::gpu::CudaEngine::global() else {
            return false;
        };
        if !cuda.can_safely_alloc(self.byte_len) {
            return false;
        }
        if let Some(buf) = cuda.try_upload_weight(self.bytes(), self.dtype) {
            self.gpu_buf = Some(buf);
            true
        } else {
            false
        }
    }

    pub fn evict_from_gpu(&mut self) {
        self.gpu_buf = None;
    }

    #[inline(always)]
    pub fn bytes(&self) -> &[u8] {
        &self.mmap.as_slice()[self.abs_offset..self.abs_offset + self.byte_len]
    }

    /// Henter ut gitt rad (`row_idx`) som `f32`-vektor (brukes f.eks. til ordliste-oppslag `embedding`).
    pub fn get_row_f32(&self, row_idx: usize, out: &mut [f32]) -> Result<()> {
        let cols = *self.dims.last().unwrap_or(&0);
        let row_bytes = (cols / self.dtype.block_size()) * self.dtype.type_size();
        let start = row_idx * row_bytes;
        let slice = &self.bytes()[start..start + row_bytes];
        dequantize_slice(slice, self.dtype, out)
    }

    /// Egenutviklet flertrådet (`rayon`) + AVX2/FMA matrisemultiplikasjon: `Y[m, n] = X[m, k] * W[n, k]^T`.
    /// Under prefill (`m > 1`) kvantiseres alle `m` rader av `X` én gang, og hver vektrad i `W`
    /// leses kun ÉN gang fra minnekartet inn i L1-cache for alle `m` tokens.
    pub fn matmul_batched(&self, x: &[f32], m: usize, k: usize) -> Result<Vec<f32>> {
        let n = self.dims[0];
        let w_k = self.dims[1];
        if k != w_k {
            bail!(
                "Dimensjonsfeil i matmul_batched: X har k={}, men W har [{}, {}]",
                k,
                n,
                w_k
            );
        }
        if let Some(ref gpu_buf) = self.gpu_buf {
            if let Some(cuda) = crate::gpu::CudaEngine::global() {
                if let Some(gpu_out) = cuda.matmul_batched(gpu_buf, x, m, n, k) {
                    return Ok(gpu_out);
                }
            }
        }

        let w_bytes = self.bytes();
        let row_bytes = (k / self.dtype.block_size()) * self.dtype.type_size();

        let mut out = vec![0.0f32; m * n];

        match self.dtype {
            GgmlDType::Q4K | GgmlDType::Q6K | GgmlDType::Q5K => {
                let nb = k / 256;
                let mut x_q8k = vec![BlockQ8K:: zeroed(); m * nb];
                for m_idx in 0..m {
                    quantize_row_q8_k(&x[m_idx * k..(m_idx + 1) * k], &mut x_q8k[m_idx * nb..(m_idx + 1) * nb]);
                }

                if m == 1 {
                    let x_row = &x_q8k[0..nb];
                    let dtype = self.dtype;
                    out.par_chunks_mut(64).enumerate().for_each(|(chunk_idx, out_chunk)| {
                        let base_col = chunk_idx * 64;
                        for (i, dst) in out_chunk.iter_mut().enumerate() {
                            let col = base_col + i;
                            let w_row = &w_bytes[col * row_bytes..(col + 1) * row_bytes];
                            *dst = unsafe {
                                match dtype {
                                    GgmlDType::Q4K => vec_dot_q4k_q8k_avx2(w_row, x_row),
                                    GgmlDType::Q6K => vec_dot_q6k_q8k_avx2(w_row, x_row),
                                    GgmlDType::Q5K => vec_dot_q5k_q8k_avx2(w_row, x_row),
                                    _ => unreachable!(),
                                }
                            };
                        }
                    });
                } else {
                    // For m > 1: regn ut [n, m] parallelt over n slik at hver vektrad i W kun leses ÉN gang!
                    let dtype = self.dtype;
                    let mut out_nm = vec![0.0f32; n * m];
                    out_nm
                        .par_chunks_mut(m * 32)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let base_col = chunk_idx * 32;
                            let cols_in_chunk = chunk.len() / m;
                            for c in 0..cols_in_chunk {
                                let col = base_col + c;
                                let w_row = &w_bytes[col * row_bytes..(col + 1) * row_bytes];
                                let dst_col = &mut chunk[c * m..(c + 1) * m];
                                for m_idx in 0..m {
                                    let x_row = &x_q8k[m_idx * nb..(m_idx + 1) * nb];
                                    dst_col[m_idx] = unsafe {
                                        match dtype {
                                            GgmlDType::Q4K => vec_dot_q4k_q8k_avx2(w_row, x_row),
                                            GgmlDType::Q6K => vec_dot_q6k_q8k_avx2(w_row, x_row),
                                            GgmlDType::Q5K => vec_dot_q5k_q8k_avx2(w_row, x_row),
                                            _ => unreachable!(),
                                        }
                                    };
                                }
                            }
                        });
                    for m_idx in 0..m {
                        let dst_row = &mut out[m_idx * n..(m_idx + 1) * n];
                        for col in 0..n {
                            dst_row[col] = out_nm[col * m + m_idx];
                        }
                    }
                }
            }
            GgmlDType::Q8_0 | GgmlDType::Q4_0 => {
                let nb = k / 32;
                let mut x_q8_0 = vec![BlockQ8_0::zeroed(); m * nb];
                for m_idx in 0..m {
                    quantize_row_q8_0(&x[m_idx * k..(m_idx + 1) * k], &mut x_q8_0[m_idx * nb..(m_idx + 1) * nb]);
                }
                let dtype = self.dtype;
                if m == 1 {
                    let x_row = &x_q8_0[0..nb];
                    out.par_chunks_mut(64).enumerate().for_each(|(chunk_idx, out_chunk)| {
                        let base_col = chunk_idx * 64;
                        for (i, dst) in out_chunk.iter_mut().enumerate() {
                            let col = base_col + i;
                            let w_row = &w_bytes[col * row_bytes..(col + 1) * row_bytes];
                            *dst = unsafe {
                                match dtype {
                                    GgmlDType::Q8_0 => vec_dot_q8_0_q8_0_avx2(w_row, x_row),
                                    GgmlDType::Q4_0 => vec_dot_q4_0_q8_0_avx2(w_row, x_row),
                                    _ => unreachable!(),
                                }
                            };
                        }
                    });
                } else {
                    let mut out_nm = vec![0.0f32; n * m];
                    out_nm
                        .par_chunks_mut(m * 32)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let base_col = chunk_idx * 32;
                            let cols_in_chunk = chunk.len() / m;
                            for c in 0..cols_in_chunk {
                                let col = base_col + c;
                                let w_row = &w_bytes[col * row_bytes..(col + 1) * row_bytes];
                                let dst_col = &mut chunk[c * m..(c + 1) * m];
                                for m_idx in 0..m {
                                    let x_row = &x_q8_0[m_idx * nb..(m_idx + 1) * nb];
                                    dst_col[m_idx] = unsafe {
                                        match dtype {
                                            GgmlDType::Q8_0 => vec_dot_q8_0_q8_0_avx2(w_row, x_row),
                                            GgmlDType::Q4_0 => vec_dot_q4_0_q8_0_avx2(w_row, x_row),
                                            _ => unreachable!(),
                                        }
                                    };
                                }
                            }
                        });
                    for m_idx in 0..m {
                        let dst_row = &mut out[m_idx * n..(m_idx + 1) * n];
                        for col in 0..n {
                            dst_row[col] = out_nm[col * m + m_idx];
                        }
                    }
                }
            }
            _ => {
                let dtype = self.dtype;
                let mut out_nm = vec![0.0f32; n * m];
                out_nm
                    .par_chunks_mut(m * 32)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let base_col = chunk_idx * 32;
                        let cols_in_chunk = chunk.len() / m;
                        let mut w_f32 = vec![0.0f32; k];
                        for c in 0..cols_in_chunk {
                            let col = base_col + c;
                            let w_row = &w_bytes[col * row_bytes..(col + 1) * row_bytes];
                            let _ = dequantize_slice(w_row, dtype, &mut w_f32);
                            let dst_col = &mut chunk[c * m..(c + 1) * m];
                            for m_idx in 0..m {
                                let x_row = &x[m_idx * k..(m_idx + 1) * k];
                                dst_col[m_idx] = unsafe { vec_dot_f32_avx2(&w_f32, x_row) };
                            }
                        }
                    });
                for m_idx in 0..m {
                    let dst_row = &mut out[m_idx * n..(m_idx + 1) * n];
                    for col in 0..n {
                        dst_row[col] = out_nm[col * m + m_idx];
                    }
                }
            }
        }

        Ok(out)
    }

    pub fn matmul_triple_same_input(
        w1: &Self,
        w2: &Self,
        w3: &Self,
        x: &[f32],
        m: usize,
        k: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if let (Some(ref b1), Some(ref b2), Some(ref b3)) = (&w1.gpu_buf, &w2.gpu_buf, &w3.gpu_buf)
        {
            if let Some(cuda) = crate::gpu::CudaEngine::global() {
                let targets = [
                    (b1.as_ref(), w1.dims[0]),
                    (b2.as_ref(), w2.dims[0]),
                    (b3.as_ref(), w3.dims[0]),
                ];
                if let Some(mut res) = cuda.matmul_group_same_input(&targets, x, m, k) {
                    if res.len() == 3 {
                        let r3 = res.pop().unwrap();
                        let r2 = res.pop().unwrap();
                        let r1 = res.pop().unwrap();
                        return Ok((r1, r2, r3));
                    }
                }
            }
        }
        Ok((
            w1.matmul_batched(x, m, k)?,
            w2.matmul_batched(x, m, k)?,
            w3.matmul_batched(x, m, k)?,
        ))
    }

    pub fn matmul_quad_same_input(
        w1: &Self,
        w2: &Self,
        w3: &Self,
        w4: &Self,
        x: &[f32],
        m: usize,
        k: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        if let (Some(ref b1), Some(ref b2), Some(ref b3), Some(ref b4)) =
            (&w1.gpu_buf, &w2.gpu_buf, &w3.gpu_buf, &w4.gpu_buf)
        {
            if let Some(cuda) = crate::gpu::CudaEngine::global() {
                let targets = [
                    (b1.as_ref(), w1.dims[0]),
                    (b2.as_ref(), w2.dims[0]),
                    (b3.as_ref(), w3.dims[0]),
                    (b4.as_ref(), w4.dims[0]),
                ];
                if let Some(mut res) = cuda.matmul_group_same_input(&targets, x, m, k) {
                    if res.len() == 4 {
                        let r4 = res.pop().unwrap();
                        let r3 = res.pop().unwrap();
                        let r2 = res.pop().unwrap();
                        let r1 = res.pop().unwrap();
                        return Ok((r1, r2, r3, r4));
                    }
                }
            }
        }
        Ok((
            w1.matmul_batched(x, m, k)?,
            w2.matmul_batched(x, m, k)?,
            w3.matmul_batched(x, m, k)?,
            w4.matmul_batched(x, m, k)?,
        ))
    }

    pub fn ffn_swiglu_batched(
        w_gate: &Self,
        w_up: &Self,
        w_down: &Self,
        ffn_in: &[f32],
        m: usize,
        n_embd: usize,
    ) -> Result<Vec<f32>> {
        if let (Some(ref b_gate), Some(ref b_up), Some(ref b_down)) =
            (&w_gate.gpu_buf, &w_up.gpu_buf, &w_down.gpu_buf)
        {
            if let Some(cuda) = crate::gpu::CudaEngine::global() {
                let ffn_inter = w_gate.dims[0];
                if let Some(down) =
                    cuda.ffn_swiglu_fused(b_gate, b_up, b_down, ffn_in, m, n_embd, ffn_inter)
                {
                    return Ok(down);
                }
            }
        }
        let mut gate = w_gate.matmul_batched(ffn_in, m, n_embd)?;
        let up = w_up.matmul_batched(ffn_in, m, n_embd)?;
        let ffn_inter = gate.len() / m;
        for i in 0..gate.len() {
            gate[i] = silu(gate[i]) * up[i];
        }
        w_down.matmul_batched(&gate, m, ffn_inter)
    }
}

// ============================================================================
// 3. EGNE KVANTISERINGSBLOKKER OG AVX2 + FMA SIMD-KJERNER
// ============================================================================

#[inline(always)]
fn f16_to_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x03ff) as u32;
    if e == 0 {
        if m == 0 {
            f32::from_bits(s)
        } else {
            let mut m = m;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            f32::from_bits(s | (e << 23) | (m << 13))
        }
    } else if e == 31 {
        f32::from_bits(s | 0x7f800000 | (m << 13))
    } else {
        f32::from_bits(s | ((e + 112) << 23) | (m << 13))
    }
}

#[inline(always)]
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[derive(Clone, Copy)]
#[repr(C)]
struct BlockQ8K {
    d: f32,
    qs: [i8; 256],
    bsums: [i16; 16],
}

impl BlockQ8K {
    #[inline(always)]
    const fn zeroed() -> Self {
        Self {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct BlockQ8_0 {
    d: f32,
    qs: [i8; 32],
}

impl BlockQ8_0 {
    #[inline(always)]
    const fn zeroed() -> Self {
        Self {
            d: 0.0,
            qs: [0i8; 32],
        }
    }
}

fn quantize_row_q8_k(x: &[f32], out: &mut [BlockQ8K]) {
    for (blk_idx, blk) in out.iter_mut().enumerate() {
        let src = &x[blk_idx * 256..(blk_idx + 1) * 256];
        let mut max_abs = 0.0f32;
        let mut max_val = 0.0f32;
        for &v in src {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
                max_val = v;
            }
        }
        if max_abs == 0.0 {
            *blk = BlockQ8K::zeroed();
            continue;
        }
        let iscale = -127.0f32 / max_val;
        blk.d = 1.0f32 / iscale;
        for j in 0..16 {
            let mut bsum = 0i16;
            let base = j * 16;
            for l in 0..16 {
                let q = (src[base + l] * iscale).round().clamp(-127.0, 127.0) as i8;
                blk.qs[base + l] = q;
                bsum += q as i16;
            }
            blk.bsums[j] = bsum;
        }
    }
}

fn quantize_row_q8_0(x: &[f32], out: &mut [BlockQ8_0]) {
    for (blk_idx, blk) in out.iter_mut().enumerate() {
        let src = &x[blk_idx * 32..(blk_idx + 1) * 32];
        let mut max_abs = 0.0f32;
        for &v in src {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs == 0.0 {
            *blk = BlockQ8_0::zeroed();
            continue;
        }
        let d = max_abs / 127.0f32;
        let id = 1.0f32 / d;
        blk.d = d;
        for i in 0..32 {
            blk.qs[i] = (src[i] * id).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

#[inline(always)]
fn decode_q4k_scales(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..4 {
        sc[j] = scales[j] & 63;
        m[j] = scales[j + 4] & 63;
    }
    for j in 4..8 {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
    (sc, m)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_epi32_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0b01_01_01_01);
    _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_ps_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// AVX2 + FMA prikkprodukt mellom `Q4_K`-rad (144 bytes per 256 vekter) og `BlockQ8K`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4k_q8k_avx2(w_bytes: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;
    let mask_0f = _mm256_set1_epi8(0x0F);
    let ones_16 = _mm256_set1_epi16(1);
    let mut total_sum = 0.0f32;

    for (blk_idx, x_blk) in x_blocks.iter().enumerate() {
        if x_blk.d == 0.0 {
            continue;
        }
        let b = &w_bytes[blk_idx * 144..(blk_idx + 1) * 144];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]])) * x_blk.d;
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]])) * x_blk.d;
        let (sc, m) = decode_q4k_scales(&b[4..16]);
        let qs_ptr = b.as_ptr().add(16);
        let x_ptr = x_blk.qs.as_ptr();

        let mut isum = 0i32;
        let mut msum = 0i32;

        for chunk in 0..4 {
            let is = chunk * 2;
            let q_bytes = _mm256_loadu_si256(qs_ptr.add(chunk * 32) as *const __m256i);
            let q_lo = _mm256_and_si256(q_bytes, mask_0f);
            let q_hi = _mm256_and_si256(_mm256_srli_epi16(q_bytes, 4), mask_0f);

            let x_lo = _mm256_loadu_si256(x_ptr.add(chunk * 64) as *const __m256i);
            let x_hi = _mm256_loadu_si256(x_ptr.add(chunk * 64 + 32) as *const __m256i);

            let prod_lo_16 = _mm256_maddubs_epi16(q_lo, x_lo);
            let prod_hi_16 = _mm256_maddubs_epi16(q_hi, x_hi);

            let sum_lo_32 = hsum_epi32_avx2(_mm256_madd_epi16(prod_lo_16, ones_16));
            let sum_hi_32 = hsum_epi32_avx2(_mm256_madd_epi16(prod_hi_16, ones_16));

            isum += sum_lo_32 * (sc[is] as i32) + sum_hi_32 * (sc[is + 1] as i32);

            let bsum_lo = (x_blk.bsums[chunk * 4] as i32) + (x_blk.bsums[chunk * 4 + 1] as i32);
            let bsum_hi =
                (x_blk.bsums[chunk * 4 + 2] as i32) + (x_blk.bsums[chunk * 4 + 3] as i32);
            msum += bsum_lo * (m[is] as i32) + bsum_hi * (m[is + 1] as i32);
        }

        total_sum += d * (isum as f32) - dmin * (msum as f32);
    }

    total_sum
}

/// AVX2 + FMA prikkprodukt mellom `Q6_K`-rad (210 bytes per 256 vekter) og `BlockQ8K`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q6k_q8k_avx2(w_bytes: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;
    let mask_0f = _mm256_set1_epi8(0x0F);
    let mask_30 = _mm256_set1_epi8(0x30);
    let ones_16 = _mm256_set1_epi16(1);
    let mut total_sum = 0.0f32;

    #[inline(always)]
    unsafe fn hsum_128_epi32(v: __m128i) -> i32 {
        let hi64 = _mm_unpackhi_epi64(v, v);
        let sum64 = _mm_add_epi32(v, hi64);
        let hi32 = _mm_shuffle_epi32(sum64, 0b01_01_01_01);
        _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32))
    }

    for (blk_idx, x_blk) in x_blocks.iter().enumerate() {
        if x_blk.d == 0.0 {
            continue;
        }
        let b = &w_bytes[blk_idx * 210..(blk_idx + 1) * 210];
        let ql_ptr = b.as_ptr();
        let qh_ptr = b.as_ptr().add(128);
        let sc_ptr = b.as_ptr().add(192) as *const i8;
        let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]])) * x_blk.d;
        let x_ptr = x_blk.qs.as_ptr();

        let mut isum = 0i32;

        for half in 0..2 {
            let ql0 = _mm256_loadu_si256(ql_ptr.add(half * 64) as *const __m256i);
            let ql1 = _mm256_loadu_si256(ql_ptr.add(half * 64 + 32) as *const __m256i);
            let qh0 = _mm256_loadu_si256(qh_ptr.add(half * 32) as *const __m256i);

            let q1_u = _mm256_or_si256(
                _mm256_and_si256(ql0, mask_0f),
                _mm256_and_si256(_mm256_slli_epi16(qh0, 4), mask_30),
            );
            let q2_u = _mm256_or_si256(
                _mm256_and_si256(ql1, mask_0f),
                _mm256_and_si256(_mm256_slli_epi16(qh0, 2), mask_30),
            );
            let q3_u = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_0f),
                _mm256_and_si256(qh0, mask_30),
            );
            let q4_u = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_0f),
                _mm256_and_si256(_mm256_srli_epi16(qh0, 2), mask_30),
            );

            let x1 = _mm256_loadu_si256(x_ptr.add(half * 128) as *const __m256i);
            let x2 = _mm256_loadu_si256(x_ptr.add(half * 128 + 32) as *const __m256i);
            let x3 = _mm256_loadu_si256(x_ptr.add(half * 128 + 64) as *const __m256i);
            let x4 = _mm256_loadu_si256(x_ptr.add(half * 128 + 96) as *const __m256i);

            let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q1_u, x1), ones_16);
            let p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2_u, x2), ones_16);
            let p3 = _mm256_madd_epi16(_mm256_maddubs_epi16(q3_u, x3), ones_16);
            let p4 = _mm256_madd_epi16(_mm256_maddubs_epi16(q4_u, x4), ones_16);

            let g_base = half * 8;
            let s0 = *sc_ptr.add(g_base) as i32;
            let s1 = *sc_ptr.add(g_base + 1) as i32;
            let s2 = *sc_ptr.add(g_base + 2) as i32;
            let s3 = *sc_ptr.add(g_base + 3) as i32;
            let s4 = *sc_ptr.add(g_base + 4) as i32;
            let s5 = *sc_ptr.add(g_base + 5) as i32;
            let s6 = *sc_ptr.add(g_base + 6) as i32;
            let s7 = *sc_ptr.add(g_base + 7) as i32;

            let d0 = hsum_128_epi32(_mm256_castsi256_si128(p1))
                - 32 * (x_blk.bsums[g_base] as i32);
            let d1 = hsum_128_epi32(_mm256_extracti128_si256(p1, 1))
                - 32 * (x_blk.bsums[g_base + 1] as i32);
            let d2 = hsum_128_epi32(_mm256_castsi256_si128(p2))
                - 32 * (x_blk.bsums[g_base + 2] as i32);
            let d3 = hsum_128_epi32(_mm256_extracti128_si256(p2, 1))
                - 32 * (x_blk.bsums[g_base + 3] as i32);
            let d4 = hsum_128_epi32(_mm256_castsi256_si128(p3))
                - 32 * (x_blk.bsums[g_base + 4] as i32);
            let d5 = hsum_128_epi32(_mm256_extracti128_si256(p3, 1))
                - 32 * (x_blk.bsums[g_base + 5] as i32);
            let d6 = hsum_128_epi32(_mm256_castsi256_si128(p4))
                - 32 * (x_blk.bsums[g_base + 6] as i32);
            let d7 = hsum_128_epi32(_mm256_extracti128_si256(p4, 1))
                - 32 * (x_blk.bsums[g_base + 7] as i32);

            isum += d0 * s0
                + d1 * s1
                + d2 * s2
                + d3 * s3
                + d4 * s4
                + d5 * s5
                + d6 * s6
                + d7 * s7;
        }

        total_sum += d * (isum as f32);
    }

    total_sum
}

/// AVX2 + FMA prikkprodukt mellom `Q5_K`-rad (176 bytes per 256 vekter) og `BlockQ8K`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q5k_q8k_avx2(w_bytes: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    let mut total_sum = 0.0f32;
    for (blk_idx, x_blk) in x_blocks.iter().enumerate() {
        if x_blk.d == 0.0 {
            continue;
        }
        let b = &w_bytes[blk_idx * 176..(blk_idx + 1) * 176];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]])) * x_blk.d;
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]])) * x_blk.d;
        let (sc, m) = decode_q4k_scales(&b[4..16]);
        let qh = &b[16..48];
        let qs = &b[48..176];

        let mut isum = 0i32;
        let mut msum = 0i32;
        let mut u1 = 1u8;
        let mut u2 = 2u8;

        for chunk in 0..4 {
            let is = chunk * 2;
            let q = &qs[chunk * 32..(chunk + 1) * 32];
            let x0 = &x_blk.qs[chunk * 64..chunk * 64 + 32];
            let x1 = &x_blk.qs[chunk * 64 + 32..chunk * 64 + 64];
            let mut s0 = 0i32;
            let mut s1 = 0i32;
            for l in 0..32 {
                let v0 = (q[l] & 0x0F) | if (qh[l] & u1) != 0 { 16 } else { 0 };
                let v1 = (q[l] >> 4) | if (qh[l] & u2) != 0 { 16 } else { 0 };
                s0 += (v0 as i32) * (x0[l] as i32);
                s1 += (v1 as i32) * (x1[l] as i32);
            }
            isum += s0 * (sc[is] as i32) + s1 * (sc[is + 1] as i32);
            let bsum_lo = (x_blk.bsums[chunk * 4] as i32) + (x_blk.bsums[chunk * 4 + 1] as i32);
            let bsum_hi =
                (x_blk.bsums[chunk * 4 + 2] as i32) + (x_blk.bsums[chunk * 4 + 3] as i32);
            msum += bsum_lo * (m[is] as i32) + bsum_hi * (m[is + 1] as i32);
            u1 = u1.wrapping_shl(2);
            u2 = u2.wrapping_shl(2);
        }

        total_sum += d * (isum as f32) - dmin * (msum as f32);
    }
    total_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q8_0_q8_0_avx2(w_bytes: &[u8], x_blocks: &[BlockQ8_0]) -> f32 {
    use std::arch::x86_64::*;
    let ones_16 = _mm256_set1_epi16(1);
    let mut total = 0.0f32;
    for (blk_idx, x_blk) in x_blocks.iter().enumerate() {
        if x_blk.d == 0.0 {
            continue;
        }
        let b = &w_bytes[blk_idx * 34..(blk_idx + 1) * 34];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]])) * x_blk.d;
        let w_vec = _mm256_loadu_si256(b.as_ptr().add(2) as *const __m256i);
        let x_vec = _mm256_loadu_si256(x_blk.qs.as_ptr() as *const __m256i);
        // Signed i8 * i8 via sign-trick for maddubs_epi16
        let w_abs = _mm256_abs_epi8(w_vec);
        let x_signed = _mm256_sign_epi8(x_vec, w_vec);
        let prod_16 = _mm256_maddubs_epi16(w_abs, x_signed);
        let isum = hsum_epi32_avx2(_mm256_madd_epi16(prod_16, ones_16));
        total += d * (isum as f32);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_0_q8_0_avx2(w_bytes: &[u8], x_blocks: &[BlockQ8_0]) -> f32 {
    let mut total = 0.0f32;
    for (blk_idx, x_blk) in x_blocks.iter().enumerate() {
        if x_blk.d == 0.0 {
            continue;
        }
        let b = &w_bytes[blk_idx * 18..(blk_idx + 1) * 18];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]])) * x_blk.d;
        let qs = &b[2..18];
        let mut isum = 0i32;
        for i in 0..16 {
            let v0 = (qs[i] & 0x0F) as i32 - 8;
            let v1 = (qs[i] >> 4) as i32 - 8;
            isum += v0 * (x_blk.qs[i] as i32) + v1 * (x_blk.qs[i + 16] as i32);
        }
        total += d * (isum as f32);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    while i + 16 <= n {
        let va0 = _mm256_loadu_ps(a_ptr.add(i));
        let vb0 = _mm256_loadu_ps(b_ptr.add(i));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        let va1 = _mm256_loadu_ps(a_ptr.add(i + 8));
        let vb1 = _mm256_loadu_ps(b_ptr.add(i + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        i += 16;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a_ptr.add(i));
        let vb = _mm256_loadu_ps(b_ptr.add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }
    let mut sum = hsum_ps_avx2(acc0);
    while i < n {
        sum += (*a_ptr.add(i)) * (*b_ptr.add(i));
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_axpy_f32_avx2(dst: &mut [f32], scale: f32, src: &[f32]) {
    use std::arch::x86_64::*;
    let n = dst.len().min(src.len());
    let s_vec = _mm256_set1_ps(scale);
    let mut i = 0;
    let d_ptr = dst.as_mut_ptr();
    let s_ptr = src.as_ptr();
    while i + 8 <= n {
        let vd = _mm256_loadu_ps(d_ptr.add(i));
        let vs = _mm256_loadu_ps(s_ptr.add(i));
        _mm256_storeu_ps(d_ptr.add(i), _mm256_fmadd_ps(s_vec, vs, vd));
        i += 8;
    }
    while i < n {
        *d_ptr.add(i) += scale * (*s_ptr.add(i));
        i += 1;
    }
}

/// Dekvantiserer en rå byte-slice (`F32`, `F16`, `BF16`, `Q4_K`, `Q6_K`, `Q5_K`, `Q8_0`, `Q4_0`) til `f32`.
fn dequantize_slice(bytes: &[u8], dtype: GgmlDType, out: &mut [f32]) -> Result<()> {
    match dtype {
        GgmlDType::F32 => {
            for (i, dst) in out.iter_mut().enumerate() {
                let b = &bytes[i * 4..i * 4 + 4];
                *dst = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        GgmlDType::F16 => {
            for (i, dst) in out.iter_mut().enumerate() {
                let b = &bytes[i * 2..i * 2 + 2];
                *dst = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        GgmlDType::BF16 => {
            for (i, dst) in out.iter_mut().enumerate() {
                let b = &bytes[i * 2..i * 2 + 2];
                *dst = bf16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        GgmlDType::Q4K => {
            let nb = out.len() / 256;
            for blk in 0..nb {
                let b = &bytes[blk * 144..(blk + 1) * 144];
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
                let (sc, m) = decode_q4k_scales(&b[4..16]);
                let qs = &b[16..144];
                let dst_blk = &mut out[blk * 256..(blk + 1) * 256];
                for chunk in 0..4 {
                    let is = chunk * 2;
                    let d1 = d * (sc[is] as f32);
                    let m1 = dmin * (m[is] as f32);
                    let d2 = d * (sc[is + 1] as f32);
                    let m2 = dmin * (m[is + 1] as f32);
                    let q = &qs[chunk * 32..(chunk + 1) * 32];
                    for l in 0..32 {
                        dst_blk[chunk * 64 + l] = d1 * ((q[l] & 0x0F) as f32) - m1;
                        dst_blk[chunk * 64 + 32 + l] = d2 * ((q[l] >> 4) as f32) - m2;
                    }
                }
            }
        }
        GgmlDType::Q6K => {
            let nb = out.len() / 256;
            for blk in 0..nb {
                let b = &bytes[blk * 210..(blk + 1) * 210];
                let ql = &b[0..128];
                let qh = &b[128..192];
                let sc = &b[192..208];
                let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
                let dst_blk = &mut out[blk * 256..(blk + 1) * 256];
                for half in 0..2 {
                    let ql_h = &ql[half * 64..(half + 1) * 64];
                    let qh_h = &qh[half * 32..(half + 1) * 32];
                    let sc_h = &sc[half * 8..(half + 1) * 8];
                    let dst_h = &mut dst_blk[half * 128..(half + 1) * 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql_h[l] & 0x0F) | (((qh_h[l] >> 0) & 3) << 4)) as i8 - 32;
                        let q2 = ((ql_h[l + 32] & 0x0F) | (((qh_h[l] >> 2) & 3) << 4)) as i8 - 32;
                        let q3 = ((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) as i8 - 32;
                        let q4 = ((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) as i8 - 32;
                        dst_h[l] = d * ((sc_h[is] as i8) as f32) * (q1 as f32);
                        dst_h[l + 32] = d * ((sc_h[is + 2] as i8) as f32) * (q2 as f32);
                        dst_h[l + 64] = d * ((sc_h[is + 4] as i8) as f32) * (q3 as f32);
                        dst_h[l + 96] = d * ((sc_h[is + 6] as i8) as f32) * (q4 as f32);
                    }
                }
            }
        }
        GgmlDType::Q5K => {
            let nb = out.len() / 256;
            for blk in 0..nb {
                let b = &bytes[blk * 176..(blk + 1) * 176];
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
                let (sc, m) = decode_q4k_scales(&b[4..16]);
                let qh = &b[16..48];
                let qs = &b[48..176];
                let dst_blk = &mut out[blk * 256..(blk + 1) * 256];
                let mut u1 = 1u8;
                let mut u2 = 2u8;
                for chunk in 0..4 {
                    let is = chunk * 2;
                    let d1 = d * (sc[is] as f32);
                    let m1 = dmin * (m[is] as f32);
                    let d2 = d * (sc[is + 1] as f32);
                    let m2 = dmin * (m[is + 1] as f32);
                    let q = &qs[chunk * 32..(chunk + 1) * 32];
                    for l in 0..32 {
                        let v0 = (q[l] & 0x0F) | if (qh[l] & u1) != 0 { 16 } else { 0 };
                        let v1 = (q[l] >> 4) | if (qh[l] & u2) != 0 { 16 } else { 0 };
                        dst_blk[chunk * 64 + l] = d1 * (v0 as f32) - m1;
                        dst_blk[chunk * 64 + 32 + l] = d2 * (v1 as f32) - m2;
                    }
                    u1 = u1.wrapping_shl(2);
                    u2 = u2.wrapping_shl(2);
                }
            }
        }
        GgmlDType::Q8_0 => {
            let nb = out.len() / 32;
            for blk in 0..nb {
                let b = &bytes[blk * 34..(blk + 1) * 34];
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                let dst_blk = &mut out[blk * 32..(blk + 1) * 32];
                for i in 0..32 {
                    dst_blk[i] = d * ((b[2 + i] as i8) as f32);
                }
            }
        }
        GgmlDType::Q4_0 => {
            let nb = out.len() / 32;
            for blk in 0..nb {
                let b = &bytes[blk * 18..(blk + 1) * 18];
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                let dst_blk = &mut out[blk * 32..(blk + 1) * 32];
                for i in 0..16 {
                    dst_blk[i] = d * (((b[2 + i] & 0x0F) as i32 - 8) as f32);
                    dst_blk[i + 16] = d * (((b[2 + i] >> 4) as i32 - 8) as f32);
                }
            }
        }
        other => bail!("Dekvantisering ikke implementert for {:?}", other),
    }
    Ok(())
}

// ============================================================================
// 4. EGNE MATTE- OG NEVRALE OPERASJONER (RMSNorm, L2Norm, SiLU, SSM AVX2)
// ============================================================================

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn rms_norm_rows(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = weight.len();
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let ss = unsafe { vec_dot_f32_avx2(row_in, row_in) };
        let inv = 1.0f32 / ((ss / (dim as f32)) + eps).sqrt();
        for i in 0..dim {
            row_out[i] = row_in[i] * inv * weight[i];
        }
    }
    out
}

fn rms_norm_inplace(buf: &mut [f32], weight: &[f32], eps: f32) {
    let dim = weight.len();
    for chunk in buf.chunks_exact_mut(dim) {
        let ss = unsafe { vec_dot_f32_avx2(chunk, chunk) };
        let inv = 1.0f32 / ((ss / (dim as f32)) + eps).sqrt();
        for i in 0..dim {
            chunk[i] = chunk[i] * inv * weight[i];
        }
    }
}

fn l2_norm_inplace(buf: &mut [f32], head_dim: usize, eps: f32) {
    for chunk in buf.chunks_exact_mut(head_dim) {
        let ss = unsafe { vec_dot_f32_avx2(chunk, chunk) };
        let inv = 1.0f32 / (ss + eps).sqrt();
        for v in chunk.iter_mut() {
            *v *= inv;
        }
    }
}

/// Fused AVX2 + FMA Gated DeltaNet SSM oppdatering for ett V-hode (`[d_v, d_k] = [128, 128]`).
/// Gjør både `sk = S * k`, `d = (v - g * sk) * beta`, `S = g * S + d * k^T` og `o = S * q`
/// i én enkelt register-pass per rad av `S`!
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ssm_head_step_avx2(
    s_head: &mut [f32],
    q_head: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    g_exp: f32,
    beta: f32,
    scale: f32,
    d_k: usize,
    d_v: usize,
    out_head: &mut [f32],
) {
    use std::arch::x86_64::*;
    let g_vec = _mm256_set1_ps(g_exp);

    for v_idx in 0..d_v {
        let s_row = &mut s_head[v_idx * d_k..(v_idx + 1) * d_k];
        let sk_val = vec_dot_f32_avx2(s_row, k_head);
        let d_val = (v_head[v_idx] - g_exp * sk_val) * beta;
        let d_vec = _mm256_set1_ps(d_val);

        let mut acc_o0 = _mm256_setzero_ps();
        let mut acc_o1 = _mm256_setzero_ps();
        let mut i = 0;
        let s_ptr = s_row.as_mut_ptr();
        let k_ptr = k_head.as_ptr();
        let q_ptr = q_head.as_ptr();

        while i + 16 <= d_k {
            let s0 = _mm256_loadu_ps(s_ptr.add(i));
            let k0 = _mm256_loadu_ps(k_ptr.add(i));
            let q0 = _mm256_loadu_ps(q_ptr.add(i));
            let s0_new = _mm256_fmadd_ps(d_vec, k0, _mm256_mul_ps(g_vec, s0));
            _mm256_storeu_ps(s_ptr.add(i), s0_new);
            acc_o0 = _mm256_fmadd_ps(s0_new, q0, acc_o0);

            let s1 = _mm256_loadu_ps(s_ptr.add(i + 8));
            let k1 = _mm256_loadu_ps(k_ptr.add(i + 8));
            let q1 = _mm256_loadu_ps(q_ptr.add(i + 8));
            let s1_new = _mm256_fmadd_ps(d_vec, k1, _mm256_mul_ps(g_vec, s1));
            _mm256_storeu_ps(s_ptr.add(i + 8), s1_new);
            acc_o1 = _mm256_fmadd_ps(s1_new, q1, acc_o1);

            i += 16;
        }
        let sum_o = hsum_ps_avx2(_mm256_add_ps(acc_o0, acc_o1));
        out_head[v_idx] = sum_o * scale;
    }
}

// ============================================================================
// 5. EGEN TOP-P / TEMPERATUR SAMPLER (XORSHIFT64 PRNG)
// ============================================================================

pub struct SelfMadeSampler {
    rng_state: u64,
    temperature: f32,
    top_p: f32,
}

impl SelfMadeSampler {
    pub fn new(seed: u64, temperature: f32, top_p: f32) -> Self {
        Self {
            rng_state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
            temperature,
            top_p,
        }
    }

    #[inline(always)]
    fn next_f32(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        ((x >> 40) as u32 as f32) / ((1u32 << 24) as f32)
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 1e-5 {
            let mut best_id = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_id = i;
                }
            }
            return best_id as u32;
        }

        let inv_temp = 1.0f32 / self.temperature;
        let mut max_logit = f32::NEG_INFINITY;
        for &v in logits {
            if v > max_logit {
                max_logit = v;
            }
        }

        // For store ordlister (f.eks. 248 000 tokens i Qwen3.5): filtrer ut kandidater
        // som er > 14.0 under max_logit (sannsynlighet < 1e-6) før sortering for lynrask Top-P!
        let cutoff = max_logit - (14.0f32 * self.temperature);
        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(256);
        let mut sum_prob = 0.0f32;

        for (i, &v) in logits.iter().enumerate() {
            if v >= cutoff {
                let p = ((v - max_logit) * inv_temp).exp();
                candidates.push((i as u32, p));
                sum_prob += p;
            }
        }

        if candidates.is_empty() || sum_prob <= 0.0 {
            return 0;
        }

        candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let target_mass = sum_prob * (self.top_p as f32);
        let mut cum_prob = 0.0f32;
        let mut keep_count = candidates.len();
        for (idx, &(_, p)) in candidates.iter().enumerate() {
            cum_prob += p;
            if cum_prob >= target_mass {
                keep_count = idx + 1;
                break;
            }
        }
        candidates.truncate(keep_count);

        let r = self.next_f32() * cum_prob.min(sum_prob);
        let mut acc = 0.0f32;
        for &(id, p) in &candidates {
            acc += p;
            if r <= acc {
                return id;
            }
        }
        candidates[0].0
    }
}

// ============================================================================
// 6. EGEN QWEN35 HYBRID-ARKITEKTUR (Gated DeltaNet SSM + Full Attention)
// ============================================================================

struct Qwen35AttentionLayer {
    attn_q: MmapWeight,
    attn_k: MmapWeight,
    attn_v: MmapWeight,
    attn_output: MmapWeight,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
}

struct Qwen35SsmLayer {
    attn_qkv: MmapWeight,
    attn_gate: MmapWeight,
    ssm_alpha: MmapWeight,
    ssm_beta: MmapWeight,
    ssm_out: MmapWeight,
    ssm_conv1d: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_dt_bias: Vec<f32>,
    ssm_norm: Vec<f32>,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

enum Qwen35Mixer {
    FullAttention(Qwen35AttentionLayer),
    GatedDeltaNet(Qwen35SsmLayer),
}

struct Qwen35Block {
    attn_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    mixer: Qwen35Mixer,
    ffn_gate: MmapWeight,
    ffn_up: MmapWeight,
    ffn_down: MmapWeight,
}

impl Qwen35Block {
    pub fn is_on_gpu(&self) -> bool {
        match &self.mixer {
            Qwen35Mixer::FullAttention(attn) => {
                attn.attn_q.gpu_buf.is_some() && self.ffn_gate.gpu_buf.is_some()
            }
            Qwen35Mixer::GatedDeltaNet(ssm) => {
                ssm.attn_qkv.gpu_buf.is_some() && self.ffn_gate.gpu_buf.is_some()
            }
        }
    }

    pub fn total_weight_bytes(&self) -> usize {
        let ffn = self.ffn_gate.byte_len + self.ffn_up.byte_len + self.ffn_down.byte_len;
        let mixer = match &self.mixer {
            Qwen35Mixer::FullAttention(attn) => {
                attn.attn_q.byte_len + attn.attn_k.byte_len + attn.attn_v.byte_len + attn.attn_output.byte_len
            }
            Qwen35Mixer::GatedDeltaNet(ssm) => {
                ssm.attn_qkv.byte_len + ssm.attn_gate.byte_len + ssm.ssm_alpha.byte_len + ssm.ssm_beta.byte_len + ssm.ssm_out.byte_len
            }
        };
        ffn + mixer
    }

    pub fn gpu_bytes(&self) -> usize {
        let mut sum = 0usize;
        for w in [&self.ffn_gate, &self.ffn_up, &self.ffn_down] {
            if let Some(ref b) = w.gpu_buf {
                sum += b.byte_len;
            }
        }
        match &self.mixer {
            Qwen35Mixer::FullAttention(attn) => {
                for w in [&attn.attn_q, &attn.attn_k, &attn.attn_v, &attn.attn_output] {
                    if let Some(ref b) = w.gpu_buf {
                        sum += b.byte_len;
                    }
                }
            }
            Qwen35Mixer::GatedDeltaNet(ssm) => {
                for w in [&ssm.attn_qkv, &ssm.attn_gate, &ssm.ssm_alpha, &ssm.ssm_beta, &ssm.ssm_out] {
                    if let Some(ref b) = w.gpu_buf {
                        sum += b.byte_len;
                    }
                }
            }
        }
        sum
    }

    pub fn try_promote_to_gpu(&mut self) -> bool {
        if self.is_on_gpu() {
            return true;
        }
        let Some(cuda) = crate::gpu::CudaEngine::global() else {
            return false;
        };
        let needed = self.total_weight_bytes();
        if !cuda.can_safely_alloc(needed) {
            return false;
        }

        let b_gate = cuda.try_upload_weight(self.ffn_gate.bytes(), self.ffn_gate.dtype);
        let b_up = cuda.try_upload_weight(self.ffn_up.bytes(), self.ffn_up.dtype);
        let b_down = cuda.try_upload_weight(self.ffn_down.bytes(), self.ffn_down.dtype);

        match &mut self.mixer {
            Qwen35Mixer::FullAttention(attn) => {
                let b_q = cuda.try_upload_weight(attn.attn_q.bytes(), attn.attn_q.dtype);
                let b_k = cuda.try_upload_weight(attn.attn_k.bytes(), attn.attn_k.dtype);
                let b_v = cuda.try_upload_weight(attn.attn_v.bytes(), attn.attn_v.dtype);
                let b_out = cuda.try_upload_weight(attn.attn_output.bytes(), attn.attn_output.dtype);

                if b_q.is_some() && b_k.is_some() && b_v.is_some() && b_out.is_some()
                    && b_gate.is_some() && b_up.is_some() && b_down.is_some()
                {
                    attn.attn_q.gpu_buf = b_q;
                    attn.attn_k.gpu_buf = b_k;
                    attn.attn_v.gpu_buf = b_v;
                    attn.attn_output.gpu_buf = b_out;
                    self.ffn_gate.gpu_buf = b_gate;
                    self.ffn_up.gpu_buf = b_up;
                    self.ffn_down.gpu_buf = b_down;
                    true
                } else {
                    self.evict_from_gpu();
                    false
                }
            }
            Qwen35Mixer::GatedDeltaNet(ssm) => {
                let b_qkv = cuda.try_upload_weight(ssm.attn_qkv.bytes(), ssm.attn_qkv.dtype);
                let b_g = cuda.try_upload_weight(ssm.attn_gate.bytes(), ssm.attn_gate.dtype);
                let b_alpha = cuda.try_upload_weight(ssm.ssm_alpha.bytes(), ssm.ssm_alpha.dtype);
                let b_beta = cuda.try_upload_weight(ssm.ssm_beta.bytes(), ssm.ssm_beta.dtype);
                let b_out = cuda.try_upload_weight(ssm.ssm_out.bytes(), ssm.ssm_out.dtype);

                if b_qkv.is_some() && b_g.is_some() && b_alpha.is_some() && b_beta.is_some() && b_out.is_some()
                    && b_gate.is_some() && b_up.is_some() && b_down.is_some()
                {
                    ssm.attn_qkv.gpu_buf = b_qkv;
                    ssm.attn_gate.gpu_buf = b_g;
                    ssm.ssm_alpha.gpu_buf = b_alpha;
                    ssm.ssm_beta.gpu_buf = b_beta;
                    ssm.ssm_out.gpu_buf = b_out;
                    self.ffn_gate.gpu_buf = b_gate;
                    self.ffn_up.gpu_buf = b_up;
                    self.ffn_down.gpu_buf = b_down;
                    true
                } else {
                    self.evict_from_gpu();
                    false
                }
            }
        }
    }

    pub fn evict_from_gpu(&mut self) {
        self.ffn_gate.gpu_buf = None;
        self.ffn_up.gpu_buf = None;
        self.ffn_down.gpu_buf = None;
        match &mut self.mixer {
            Qwen35Mixer::FullAttention(attn) => {
                attn.attn_q.gpu_buf = None;
                attn.attn_k.gpu_buf = None;
                attn.attn_v.gpu_buf = None;
                attn.attn_output.gpu_buf = None;
            }
            Qwen35Mixer::GatedDeltaNet(ssm) => {
                ssm.attn_qkv.gpu_buf = None;
                ssm.attn_gate.gpu_buf = None;
                ssm.ssm_alpha.gpu_buf = None;
                ssm.ssm_beta.gpu_buf = None;
                ssm.ssm_out.gpu_buf = None;
            }
        }
    }
}

pub struct Qwen35NativeEngine {
    tok_embeddings: MmapWeight,
    output_norm: Vec<f32>,
    output: MmapWeight,
    blocks: Vec<Qwen35Block>,
    n_embd: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rms_eps: f32,
    ssm_conv_kernel: usize,
    ssm_inner_size: usize,
    ssm_state_size: usize,
    ssm_time_step_rank: usize,
    ssm_n_group: usize,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl Qwen35NativeEngine {
    pub fn from_gguf(gguf: &GgufModelFile) -> Result<Self> {
        let n_layers = gguf.get_usize("qwen35.block_count")?;
        let n_embd = gguf.get_usize("qwen35.embedding_length")?;
        let n_heads = gguf.get_usize("qwen35.attention.head_count")?;
        let n_kv_heads = gguf.get_usize("qwen35.attention.head_count_kv")?;
        let head_dim = gguf
            .get_usize("qwen35.attention.key_length")
            .unwrap_or(256);
        let rope_dim = gguf
            .get_usize("qwen35.rope.dimension_count")
            .unwrap_or(64);
        let rms_eps = gguf
            .get_f32("qwen35.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-6);
        let rope_theta = gguf
            .get_f32("qwen35.rope.freq_base")
            .unwrap_or(10_000_000.0);

        let ssm_conv_kernel = gguf.get_usize("qwen35.ssm.conv_kernel").unwrap_or(4);
        let ssm_inner_size = gguf.get_usize("qwen35.ssm.inner_size").unwrap_or(4096);
        let ssm_state_size = gguf.get_usize("qwen35.ssm.state_size").unwrap_or(128);
        let ssm_time_step_rank = gguf.get_usize("qwen35.ssm.time_step_rank").unwrap_or(32);
        let ssm_n_group = gguf.get_usize("qwen35.ssm.group_count").unwrap_or(16);

        let max_seq_len = 8192usize;
        let half_rope = rope_dim / 2;
        let mut rope_cos = vec![0.0f32; max_seq_len * half_rope];
        let mut rope_sin = vec![0.0f32; max_seq_len * half_rope];
        for pos in 0..max_seq_len {
            for j in 0..half_rope {
                let freq = 1.0f32 / rope_theta.powf((2 * j) as f32 / (rope_dim as f32));
                let angle = (pos as f32) * freq;
                rope_cos[pos * half_rope + j] = angle.cos();
                rope_sin[pos * half_rope + j] = angle.sin();
            }
        }

        let tok_embeddings = gguf.tensor_no_gpu("token_embd.weight")?;
        let output_norm = gguf.tensor_f32_1d("output_norm.weight")?;
        let output = gguf
            .tensor_no_gpu("output.weight")
            .or_else(|_| gguf.tensor_no_gpu("token_embd.weight"))?;

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let prefix = format!("blk.{}", i);
            let attn_norm = gguf.tensor_f32_1d(&format!("{}.attn_norm.weight", prefix))?;
            let post_attention_norm =
                gguf.tensor_f32_1d(&format!("{}.post_attention_norm.weight", prefix))?;
            let ffn_gate = gguf.tensor_no_gpu(&format!("{}.ffn_gate.weight", prefix))?;
            let ffn_up = gguf.tensor_no_gpu(&format!("{}.ffn_up.weight", prefix))?;
            let ffn_down = gguf.tensor_no_gpu(&format!("{}.ffn_down.weight", prefix))?;

            let has_full_attn = gguf
                .tensors
                .contains_key(&format!("{}.attn_q.weight", prefix));
            let mixer = if has_full_attn {
                let attn_q = gguf.tensor_no_gpu(&format!("{}.attn_q.weight", prefix))?;
                let attn_k = gguf.tensor_no_gpu(&format!("{}.attn_k.weight", prefix))?;
                let attn_v = gguf.tensor_no_gpu(&format!("{}.attn_v.weight", prefix))?;
                let attn_output = gguf.tensor_no_gpu(&format!("{}.attn_output.weight", prefix))?;
                let attn_q_norm = gguf.tensor_f32_1d(&format!("{}.attn_q_norm.weight", prefix))?;
                let attn_k_norm = gguf.tensor_f32_1d(&format!("{}.attn_k_norm.weight", prefix))?;
                Qwen35Mixer::FullAttention(Qwen35AttentionLayer {
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                    k_cache: Vec::new(),
                    v_cache: Vec::new(),
                })
            } else {
                let attn_qkv = gguf.tensor_no_gpu(&format!("{}.attn_qkv.weight", prefix))?;
                let attn_gate = gguf.tensor_no_gpu(&format!("{}.attn_gate.weight", prefix))?;
                let ssm_alpha = gguf.tensor_no_gpu(&format!("{}.ssm_alpha.weight", prefix))?;
                let ssm_beta = gguf.tensor_no_gpu(&format!("{}.ssm_beta.weight", prefix))?;
                let ssm_out = gguf.tensor_no_gpu(&format!("{}.ssm_out.weight", prefix))?;
                let (ssm_conv1d, conv_dims) =
                    gguf.tensor_f32_flat(&format!("{}.ssm_conv1d.weight", prefix))?;
                let conv_channels = conv_dims[0];
                let ssm_a = gguf.tensor_f32_1d(&format!("{}.ssm_a", prefix))?;
                let ssm_dt_bias = gguf
                    .tensor_f32_1d(&format!("{}.ssm_dt.bias", prefix))
                    .or_else(|_| gguf.tensor_f32_1d(&format!("{}.ssm_dt_bias", prefix)))?;
                let ssm_norm = gguf.tensor_f32_1d(&format!("{}.ssm_norm.weight", prefix))?;
                let d_v = ssm_inner_size / ssm_time_step_rank;
                Qwen35Mixer::GatedDeltaNet(Qwen35SsmLayer {
                    attn_qkv,
                    attn_gate,
                    ssm_alpha,
                    ssm_beta,
                    ssm_out,
                    ssm_conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    conv_state: vec![0.0f32; conv_channels * (ssm_conv_kernel - 1)],
                    ssm_state: vec![0.0f32; ssm_time_step_rank * ssm_state_size * d_v],
                })
            };

            blocks.push(Qwen35Block {
                attn_norm,
                post_attention_norm,
                mixer,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let mut engine = Self {
            tok_embeddings,
            output_norm,
            output,
            blocks,
            n_embd,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dim,
            rms_eps,
            ssm_conv_kernel,
            ssm_inner_size,
            ssm_state_size,
            ssm_time_step_rank,
            ssm_n_group,
            rope_cos,
            rope_sin,
        };

        if let Some(cuda) = crate::gpu::CudaEngine::global() {
            println!("Starter dynamisk lag-allokering på GPU ({}) ...", cuda.device_name);
            for (i, blk) in engine.blocks.iter_mut().enumerate() {
                if !blk.try_promote_to_gpu() {
                    println!("GPU VRAM nådde trygg sub-maksgrense ved lag {}. Resterende lag kjøres på CPU.", i);
                    break;
                }
            }
            if engine.blocks.iter().all(|b| b.is_on_gpu()) {
                let _ = engine.output.try_promote_to_gpu();
            }
        }
        let status = engine.get_hybrid_status();
        println!("Allokeringsstatus: {}", status.summary());
        Ok(engine)
    }

    pub fn get_hybrid_status(&self) -> HybridStatus {
        let total_layers = self.blocks.len();
        let gpu_layers = self.blocks.iter().filter(|b| b.is_on_gpu()).count();
        let cpu_layers = total_layers.saturating_sub(gpu_layers);
        let mut gpu_vram_bytes: usize = self.blocks.iter().map(|b| b.gpu_bytes()).sum();
        let output_on_gpu = self.output.gpu_buf.is_some();
        if let Some(ref buf) = self.output.gpu_buf {
            gpu_vram_bytes += buf.byte_len;
        }
        HybridStatus {
            total_layers,
            gpu_layers,
            cpu_layers,
            gpu_vram_bytes,
            output_on_gpu,
        }
    }

    pub fn dynamically_adjust_placement(&mut self, snap: &crate::hardware::ResourceSnapshot) -> Option<String> {
        let Some(cuda) = crate::gpu::CudaEngine::global() else {
            return None;
        };

        if snap.level == crate::hardware::ResourceLevel::CriticalPause
            || snap.level == crate::hardware::ResourceLevel::LowResources
            || cuda.is_vram_under_pressure()
        {
            for i in (0..self.blocks.len()).rev() {
                if self.blocks[i].is_on_gpu() {
                    self.blocks[i].evict_from_gpu();
                    let st = self.get_hybrid_status();
                    let msg = format!("⚡ Dynamisk avlasting: flyttet lag {} til CPU (VRAM-trykk). Nå: {}/{} på GPU", i, st.gpu_layers, st.total_layers);
                    println!("{}", msg);
                    return Some(msg);
                }
            }
            if self.output.gpu_buf.is_some() {
                self.output.evict_from_gpu();
                return Some("⚡ Dynamisk avlasting: flyttet output-lag til CPU".to_string());
            }
        } else if snap.level == crate::hardware::ResourceLevel::OptimalSubMax
            && !cuda.is_vram_under_pressure()
        {
            for i in 0..self.blocks.len() {
                if !self.blocks[i].is_on_gpu() {
                    let needed = self.blocks[i].total_weight_bytes();
                    if cuda.can_safely_alloc(needed) {
                        if self.blocks[i].try_promote_to_gpu() {
                            let st = self.get_hybrid_status();
                            let msg = format!("⚡ Dynamisk akselerasjon: forfremmet lag {} til GPU VRAM. Nå: {}/{} på GPU", i, st.gpu_layers, st.total_layers);
                            println!("{}", msg);
                            return Some(msg);
                        }
                    }
                    break;
                }
            }
            if !self.output.gpu_buf.is_some() && cuda.can_safely_alloc(self.output.byte_len) {
                if self.output.try_promote_to_gpu() {
                    return Some("⚡ Dynamisk akselerasjon: forfremmet output-lag til GPU VRAM".to_string());
                }
            }
        }
        None
    }

    pub fn clear_state(&mut self) {
        for blk in &mut self.blocks {
            match &mut blk.mixer {
                Qwen35Mixer::FullAttention(attn) => {
                    attn.k_cache.clear();
                    attn.v_cache.clear();
                }
                Qwen35Mixer::GatedDeltaNet(ssm) => {
                    ssm.conv_state.fill(0.0);
                    ssm.ssm_state.fill(0.0);
                }
            }
        }
    }

    pub fn forward(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            bail!("Tom token-liste til Qwen35NativeEngine::forward");
        }

        let mut x = vec![0.0f32; seq_len * self.n_embd];
        for (t_idx, &tok) in tokens.iter().enumerate() {
            self.tok_embeddings.get_row_f32(
                tok as usize,
                &mut x[t_idx * self.n_embd..(t_idx + 1) * self.n_embd],
            )?;
        }

        let half_rope = self.rope_dim / 2;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;
        let rope_dim = self.rope_dim;
        let rms_eps = self.rms_eps;
        let n_embd = self.n_embd;

        let ssm_conv_kernel = self.ssm_conv_kernel;
        let ssm_inner_size = self.ssm_inner_size;
        let ssm_state_size = self.ssm_state_size;
        let ssm_time_step_rank = self.ssm_time_step_rank;
        let ssm_n_group = self.ssm_n_group;

        for blk in &mut self.blocks {
            if let Some(flag) = cancel_flag {
                if flag.load(Ordering::Relaxed) {
                    bail!("Avbrutt av bruker");
                }
            }
            let normed = rms_norm_rows(&x, &blk.attn_norm, rms_eps);

            let mixer_out = match &mut blk.mixer {
                Qwen35Mixer::FullAttention(attn) => {
                    let (q_gate_raw, mut k_raw, v_raw) = MmapWeight::matmul_triple_same_input(
                        &attn.attn_q,
                        &attn.attn_k,
                        &attn.attn_v,
                        &normed,
                        seq_len,
                        n_embd,
                    )?;

                    let mut q_only = vec![0.0f32; seq_len * n_heads * head_dim];
                    let mut gate_only = vec![0.0f32; seq_len * n_heads * head_dim];
                    for s in 0..seq_len {
                        for h in 0..n_heads {
                            let src_base = (s * n_heads + h) * (2 * head_dim);
                            let dst_base = (s * n_heads + h) * head_dim;
                            q_only[dst_base..dst_base + head_dim]
                                .copy_from_slice(&q_gate_raw[src_base..src_base + head_dim]);
                            gate_only[dst_base..dst_base + head_dim].copy_from_slice(
                                &q_gate_raw[src_base + head_dim..src_base + 2 * head_dim],
                            );
                        }
                    }

                    rms_norm_inplace(&mut q_only, &attn.attn_q_norm, rms_eps);
                    rms_norm_inplace(&mut k_raw, &attn.attn_k_norm, rms_eps);

                    for s in 0..seq_len {
                        let pos = (start_pos + s).min(8191);
                        let cos_row = &self.rope_cos[pos * half_rope..(pos + 1) * half_rope];
                        let sin_row = &self.rope_sin[pos * half_rope..(pos + 1) * half_rope];

                        for h in 0..n_heads {
                            let base = (s * n_heads + h) * head_dim;
                            for j in 0..half_rope {
                                let x0 = q_only[base + j];
                                let x1 = q_only[base + half_rope + j];
                                q_only[base + j] = x0 * cos_row[j] - x1 * sin_row[j];
                                q_only[base + half_rope + j] = x1 * cos_row[j] + x0 * sin_row[j];
                            }
                        }
                        for kh in 0..n_kv_heads {
                            let base = (s * n_kv_heads + kh) * head_dim;
                            for j in 0..half_rope {
                                let x0 = k_raw[base + j];
                                let x1 = k_raw[base + half_rope + j];
                                k_raw[base + j] = x0 * cos_row[j] - x1 * sin_row[j];
                                k_raw[base + half_rope + j] = x1 * cos_row[j] + x0 * sin_row[j];
                            }
                        }
                    }

                    attn.k_cache.extend_from_slice(&k_raw);
                    attn.v_cache.extend_from_slice(&v_raw);

                    let total_seq = attn.k_cache.len() / (n_kv_heads * head_dim);
                    let gqa_ratio = n_heads / n_kv_heads;
                    let scale = 1.0f32 / (head_dim as f32).sqrt();

                    let k_cache_ref = &attn.k_cache;
                    let v_cache_ref = &attn.v_cache;
                    let mut attn_out = vec![0.0f32; seq_len * n_heads * head_dim];

                    for s in 0..seq_len {
                        let max_t = start_pos + s + 1;
                        let valid_t = max_t.min(total_seq);
                        let q_s = &q_only[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
                        let gate_s =
                            &gate_only[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
                        let out_s =
                            &mut attn_out[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];

                        out_s
                            .par_chunks_mut(head_dim)
                            .enumerate()
                            .for_each(|(h, out_h)| {
                                let kh = h / gqa_ratio;
                                let q_h = &q_s[h * head_dim..(h + 1) * head_dim];
                                let gate_h = &gate_s[h * head_dim..(h + 1) * head_dim];
                                let mut scores = vec![0.0f32; valid_t];
                                let mut max_score = f32::NEG_INFINITY;
                                for t in 0..valid_t {
                                    let k_t = &k_cache_ref[(t * n_kv_heads + kh) * head_dim
                                        ..(t * n_kv_heads + kh + 1) * head_dim];
                                    let sc = unsafe { vec_dot_f32_avx2(q_h, k_t) } * scale;
                                    scores[t] = sc;
                                    if sc > max_score {
                                        max_score = sc;
                                    }
                                }
                                let mut sum_exp = 0.0f32;
                                for sc in &mut scores {
                                    *sc = (*sc - max_score).exp();
                                    sum_exp += *sc;
                                }
                                let inv_sum = 1.0f32 / sum_exp.max(1e-12);
                                for t in 0..valid_t {
                                    let att = scores[t] * inv_sum;
                                    let v_t = &v_cache_ref[(t * n_kv_heads + kh) * head_dim
                                        ..(t * n_kv_heads + kh + 1) * head_dim];
                                    unsafe {
                                        vec_axpy_f32_avx2(out_h, att, v_t);
                                    }
                                }
                                for d in 0..head_dim {
                                    out_h[d] *= sigmoid(gate_h[d]);
                                }
                            });
                    }

                    let _ = rope_dim;
                    attn.attn_output
                        .matmul_batched(&attn_out, seq_len, n_heads * head_dim)?
                }
                Qwen35Mixer::GatedDeltaNet(ssm) => {
                    let (qkv_raw, z_raw, alpha_raw, beta_raw) = MmapWeight::matmul_quad_same_input(
                        &ssm.attn_qkv,
                        &ssm.attn_gate,
                        &ssm.ssm_alpha,
                        &ssm.ssm_beta,
                        &normed,
                        seq_len,
                        n_embd,
                    )?;

                    let d_k = ssm_state_size;
                    let n_k_heads = ssm_n_group;
                    let n_v_heads = ssm_time_step_rank;
                    let d_v = ssm_inner_size / n_v_heads;
                    let q_dim = n_k_heads * d_k;
                    let k_dim = n_k_heads * d_k;
                    let v_dim = n_v_heads * d_v;
                    let conv_channels = q_dim + k_dim + v_dim;
                    let hist_len = ssm_conv_kernel - 1;
                    let scale = 1.0f32 / (d_k as f32).sqrt();

                    let mut y_all = vec![0.0f32; seq_len * v_dim];

                    for s in 0..seq_len {
                        let qkv_s = &qkv_raw[s * conv_channels..(s + 1) * conv_channels];
                        let z_s = &z_raw[s * v_dim..(s + 1) * v_dim];
                        let alpha_s = &alpha_raw[s * n_v_heads..(s + 1) * n_v_heads];
                        let beta_s = &beta_raw[s * n_v_heads..(s + 1) * n_v_heads];

                        let mut conv_out = vec![0.0f32; conv_channels];
                        for c in 0..conv_channels {
                            let w_base = c * ssm_conv_kernel;
                            let st_base = c * hist_len;
                            let mut acc = 0.0f32;
                            for k_idx in 0..hist_len {
                                acc += ssm.conv_state[st_base + k_idx]
                                    * ssm.ssm_conv1d[w_base + k_idx];
                            }
                            acc += qkv_s[c] * ssm.ssm_conv1d[w_base + hist_len];
                            conv_out[c] = silu(acc);

                            for k_idx in 0..(hist_len - 1) {
                                ssm.conv_state[st_base + k_idx] =
                                    ssm.conv_state[st_base + k_idx + 1];
                            }
                            ssm.conv_state[st_base + hist_len - 1] = qkv_s[c];
                        }

                        let (q_slice, rest) = conv_out.split_at_mut(q_dim);
                        let (k_slice, v_slice) = rest.split_at_mut(k_dim);

                        l2_norm_inplace(q_slice, d_k, rms_eps);
                        l2_norm_inplace(k_slice, d_k, rms_eps);

                        let y_s = &mut y_all[s * v_dim..(s + 1) * v_dim];
                        let ssm_a_ref = &ssm.ssm_a;
                        let ssm_dt_bias_ref = &ssm.ssm_dt_bias;
                        let ssm_norm_ref = &ssm.ssm_norm;
                        let q_ref = &*q_slice;
                        let k_ref = &*k_slice;
                        let v_ref = &*v_slice;

                        // Parallelliser de 32 uavhengige V-hodene over CPU-kjernene + AVX2/FMA
                        ssm.ssm_state
                            .par_chunks_mut(d_k * d_v)
                            .zip(y_s.par_chunks_mut(d_v))
                            .enumerate()
                            .for_each(|(vh, (s_head, out_head))| {
                                let kh = vh % n_k_heads;
                                let q_head = &q_ref[kh * d_k..(kh + 1) * d_k];
                                let k_head = &k_ref[kh * d_k..(kh + 1) * d_k];
                                let v_head = &v_ref[vh * d_v..(vh + 1) * d_v];
                                let z_head = &z_s[vh * d_v..(vh + 1) * d_v];

                                let beta = sigmoid(beta_s[vh]);
                                let g = softplus(alpha_s[vh] + ssm_dt_bias_ref[vh]) * ssm_a_ref[vh];
                                let g_exp = g.exp();

                                unsafe {
                                    ssm_head_step_avx2(
                                        s_head, q_head, k_head, v_head, g_exp, beta, scale, d_k,
                                        d_v, out_head,
                                    );
                                }

                                let ss = unsafe { vec_dot_f32_avx2(out_head, out_head) };
                                let inv_rms = 1.0f32 / ((ss / (d_v as f32)) + rms_eps).sqrt();
                                for v_idx in 0..d_v {
                                    out_head[v_idx] = out_head[v_idx]
                                        * inv_rms
                                        * ssm_norm_ref[v_idx]
                                        * silu(z_head[v_idx]);
                                }
                            });
                    }

                    ssm.ssm_out.matmul_batched(&y_all, seq_len, v_dim)?
                }
            };

            for i in 0..x.len() {
                x[i] += mixer_out[i];
            }

            let ffn_in = rms_norm_rows(&x, &blk.post_attention_norm, rms_eps);
            let down = MmapWeight::ffn_swiglu_batched(
                &blk.ffn_gate,
                &blk.ffn_up,
                &blk.ffn_down,
                &ffn_in,
                seq_len,
                n_embd,
            )?;
            for i in 0..x.len() {
                x[i] += down[i];
            }
        }

        let last_hidden = &x[(seq_len - 1) * n_embd..seq_len * n_embd];
        let normed_last = rms_norm_rows(last_hidden, &self.output_norm, rms_eps);
        let logits = self.output.matmul_batched(&normed_last, 1, n_embd)?;
        Ok(logits)
    }
}

// ============================================================================
// 7. EGEN STANDARD TRANSFORMER-MOTOR (Llama / Qwen2 / Mistral / Gemma / Phi3)
// ============================================================================

struct StandardBlock {
    attn_norm: Vec<f32>,
    attn_q: MmapWeight,
    attn_k: MmapWeight,
    attn_v: MmapWeight,
    attn_q_bias: Option<Vec<f32>>,
    attn_k_bias: Option<Vec<f32>>,
    attn_v_bias: Option<Vec<f32>>,
    attn_output: MmapWeight,
    ffn_norm: Vec<f32>,
    ffn_gate: MmapWeight,
    ffn_up: MmapWeight,
    ffn_down: MmapWeight,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
}

impl StandardBlock {
    pub fn is_on_gpu(&self) -> bool {
        self.attn_q.gpu_buf.is_some() && self.ffn_gate.gpu_buf.is_some()
    }

    pub fn total_weight_bytes(&self) -> usize {
        self.attn_q.byte_len + self.attn_k.byte_len + self.attn_v.byte_len +
        self.attn_output.byte_len + self.ffn_gate.byte_len + self.ffn_up.byte_len +
        self.ffn_down.byte_len
    }

    pub fn gpu_bytes(&self) -> usize {
        let mut s = 0;
        for w in [
            &self.attn_q, &self.attn_k, &self.attn_v, &self.attn_output,
            &self.ffn_gate, &self.ffn_up, &self.ffn_down
        ] {
            if let Some(ref b) = w.gpu_buf {
                s += b.byte_len;
            }
        }
        s
    }

    pub fn try_promote_to_gpu(&mut self) -> bool {
        if self.is_on_gpu() {
            return true;
        }
        let Some(cuda) = crate::gpu::CudaEngine::global() else {
            return false;
        };
        let needed = self.total_weight_bytes();
        if !cuda.can_safely_alloc(needed) {
            return false;
        }
        let b_q = cuda.try_upload_weight(self.attn_q.bytes(), self.attn_q.dtype);
        let b_k = cuda.try_upload_weight(self.attn_k.bytes(), self.attn_k.dtype);
        let b_v = cuda.try_upload_weight(self.attn_v.bytes(), self.attn_v.dtype);
        let b_out = cuda.try_upload_weight(self.attn_output.bytes(), self.attn_output.dtype);
        let b_gate = cuda.try_upload_weight(self.ffn_gate.bytes(), self.ffn_gate.dtype);
        let b_up = cuda.try_upload_weight(self.ffn_up.bytes(), self.ffn_up.dtype);
        let b_down = cuda.try_upload_weight(self.ffn_down.bytes(), self.ffn_down.dtype);

        if b_q.is_some() && b_k.is_some() && b_v.is_some() && b_out.is_some()
            && b_gate.is_some() && b_up.is_some() && b_down.is_some()
        {
            self.attn_q.gpu_buf = b_q;
            self.attn_k.gpu_buf = b_k;
            self.attn_v.gpu_buf = b_v;
            self.attn_output.gpu_buf = b_out;
            self.ffn_gate.gpu_buf = b_gate;
            self.ffn_up.gpu_buf = b_up;
            self.ffn_down.gpu_buf = b_down;
            true
        } else {
            self.evict_from_gpu();
            false
        }
    }

    pub fn evict_from_gpu(&mut self) {
        self.attn_q.gpu_buf = None;
        self.attn_k.gpu_buf = None;
        self.attn_v.gpu_buf = None;
        self.attn_output.gpu_buf = None;
        self.ffn_gate.gpu_buf = None;
        self.ffn_up.gpu_buf = None;
        self.ffn_down.gpu_buf = None;
    }
}

pub struct StandardNativeEngine {
    tok_embeddings: MmapWeight,
    output_norm: Vec<f32>,
    output: MmapWeight,
    blocks: Vec<StandardBlock>,
    n_embd: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl StandardNativeEngine {
    pub fn from_gguf(gguf: &GgufModelFile, arch: &str) -> Result<Self> {
        let prefix_arch = if gguf.metadata.contains_key(&format!("{}.block_count", arch)) {
            arch
        } else {
            "llama"
        };
        let n_layers = gguf.get_usize(&format!("{}.block_count", prefix_arch))?;
        let n_embd = gguf.get_usize(&format!("{}.embedding_length", prefix_arch))?;
        let n_heads = gguf.get_usize(&format!("{}.attention.head_count", prefix_arch))?;
        let n_kv_heads = gguf
            .get_usize(&format!("{}.attention.head_count_kv", prefix_arch))
            .unwrap_or(n_heads);
        let head_dim = gguf
            .get_usize(&format!("{}.attention.key_length", prefix_arch))
            .unwrap_or(n_embd / n_heads);
        let rms_eps = gguf
            .get_f32(&format!(
                "{}.attention.layer_norm_rms_epsilon",
                prefix_arch
            ))
            .unwrap_or(1e-5);
        let rope_theta = gguf
            .get_f32(&format!("{}.rope.freq_base", prefix_arch))
            .unwrap_or(10_000.0);

        let max_seq_len = 8192usize;
        let half_dim = head_dim / 2;
        let mut rope_cos = vec![0.0f32; max_seq_len * half_dim];
        let mut rope_sin = vec![0.0f32; max_seq_len * half_dim];
        for pos in 0..max_seq_len {
            for j in 0..half_dim {
                let freq = 1.0f32 / rope_theta.powf((2 * j) as f32 / (head_dim as f32));
                let angle = (pos as f32) * freq;
                rope_cos[pos * half_dim + j] = angle.cos();
                rope_sin[pos * half_dim + j] = angle.sin();
            }
        }

        let tok_embeddings = gguf.tensor_no_gpu("token_embd.weight")?;
        let output_norm = gguf.tensor_f32_1d("output_norm.weight")?;
        let output = gguf
            .tensor_no_gpu("output.weight")
            .or_else(|_| gguf.tensor_no_gpu("token_embd.weight"))?;

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("blk.{}", i);
            let attn_norm = gguf.tensor_f32_1d(&format!("{}.attn_norm.weight", p))?;
            let attn_q = gguf.tensor_no_gpu(&format!("{}.attn_q.weight", p))?;
            let attn_k = gguf.tensor_no_gpu(&format!("{}.attn_k.weight", p))?;
            let attn_v = gguf.tensor_no_gpu(&format!("{}.attn_v.weight", p))?;
            let attn_q_bias = gguf.tensor_f32_1d(&format!("{}.attn_q.bias", p)).ok();
            let attn_k_bias = gguf.tensor_f32_1d(&format!("{}.attn_k.bias", p)).ok();
            let attn_v_bias = gguf.tensor_f32_1d(&format!("{}.attn_v.bias", p)).ok();
            let attn_output = gguf.tensor_no_gpu(&format!("{}.attn_output.weight", p))?;
            let ffn_norm = gguf.tensor_f32_1d(&format!("{}.ffn_norm.weight", p))?;
            let ffn_gate = gguf.tensor_no_gpu(&format!("{}.ffn_gate.weight", p))?;
            let ffn_up = gguf.tensor_no_gpu(&format!("{}.ffn_up.weight", p))?;
            let ffn_down = gguf.tensor_no_gpu(&format!("{}.ffn_down.weight", p))?;

            blocks.push(StandardBlock {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
                attn_output,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                k_cache: Vec::new(),
                v_cache: Vec::new(),
            });
        }

        let mut engine = Self {
            tok_embeddings,
            output_norm,
            output,
            blocks,
            n_embd,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_eps,
            rope_cos,
            rope_sin,
        };

        if let Some(cuda) = crate::gpu::CudaEngine::global() {
            println!("Starter dynamisk lag-allokering på GPU ({}) ...", cuda.device_name);
            for (i, blk) in engine.blocks.iter_mut().enumerate() {
                if !blk.try_promote_to_gpu() {
                    println!("GPU VRAM nådde trygg sub-maksgrense ved lag {}. Resterende lag kjøres på CPU.", i);
                    break;
                }
            }
            if engine.blocks.iter().all(|b| b.is_on_gpu()) {
                let _ = engine.output.try_promote_to_gpu();
            }
        }
        let status = engine.get_hybrid_status();
        println!("Allokeringsstatus: {}", status.summary());
        Ok(engine)
    }

    pub fn get_hybrid_status(&self) -> HybridStatus {
        let total_layers = self.blocks.len();
        let gpu_layers = self.blocks.iter().filter(|b| b.is_on_gpu()).count();
        let cpu_layers = total_layers.saturating_sub(gpu_layers);
        let mut gpu_vram_bytes: usize = self.blocks.iter().map(|b| b.gpu_bytes()).sum();
        let output_on_gpu = self.output.gpu_buf.is_some();
        if let Some(ref buf) = self.output.gpu_buf {
            gpu_vram_bytes += buf.byte_len;
        }
        HybridStatus {
            total_layers,
            gpu_layers,
            cpu_layers,
            gpu_vram_bytes,
            output_on_gpu,
        }
    }

    pub fn dynamically_adjust_placement(&mut self, snap: &crate::hardware::ResourceSnapshot) -> Option<String> {
        let Some(cuda) = crate::gpu::CudaEngine::global() else {
            return None;
        };

        if snap.level == crate::hardware::ResourceLevel::CriticalPause
            || snap.level == crate::hardware::ResourceLevel::LowResources
            || cuda.is_vram_under_pressure()
        {
            for i in (0..self.blocks.len()).rev() {
                if self.blocks[i].is_on_gpu() {
                    self.blocks[i].evict_from_gpu();
                    let st = self.get_hybrid_status();
                    let msg = format!("⚡ Dynamisk avlasting: flyttet lag {} til CPU (VRAM-trykk). Nå: {}/{} på GPU", i, st.gpu_layers, st.total_layers);
                    println!("{}", msg);
                    return Some(msg);
                }
            }
            if self.output.gpu_buf.is_some() {
                self.output.evict_from_gpu();
                return Some("⚡ Dynamisk avlasting: flyttet output-lag til CPU".to_string());
            }
        } else if snap.level == crate::hardware::ResourceLevel::OptimalSubMax
            && !cuda.is_vram_under_pressure()
        {
            for i in 0..self.blocks.len() {
                if !self.blocks[i].is_on_gpu() {
                    let needed = self.blocks[i].total_weight_bytes();
                    if cuda.can_safely_alloc(needed) {
                        if self.blocks[i].try_promote_to_gpu() {
                            let st = self.get_hybrid_status();
                            let msg = format!("⚡ Dynamisk akselerasjon: forfremmet lag {} til GPU VRAM. Nå: {}/{} på GPU", i, st.gpu_layers, st.total_layers);
                            println!("{}", msg);
                            return Some(msg);
                        }
                    }
                    break;
                }
            }
            if !self.output.gpu_buf.is_some() && cuda.can_safely_alloc(self.output.byte_len) {
                if self.output.try_promote_to_gpu() {
                    return Some("⚡ Dynamisk akselerasjon: forfremmet output-lag til GPU VRAM".to_string());
                }
            }
        }
        None
    }

    pub fn clear_state(&mut self) {
        for blk in &mut self.blocks {
            blk.k_cache.clear();
            blk.v_cache.clear();
        }
    }

    pub fn forward(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let seq_len = tokens.len();
        let mut x = vec![0.0f32; seq_len * self.n_embd];
        for (t_idx, &tok) in tokens.iter().enumerate() {
            self.tok_embeddings.get_row_f32(
                tok as usize,
                &mut x[t_idx * self.n_embd..(t_idx + 1) * self.n_embd],
            )?;
        }

        let n_embd = self.n_embd;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;
        let half_dim = head_dim / 2;
        let rms_eps = self.rms_eps;

        for blk in &mut self.blocks {
            if let Some(flag) = cancel_flag {
                if flag.load(Ordering::Relaxed) {
                    bail!("Avbrutt av bruker");
                }
            }
            let normed = rms_norm_rows(&x, &blk.attn_norm, rms_eps);
            let (mut q, mut k, mut v) = MmapWeight::matmul_triple_same_input(
                &blk.attn_q,
                &blk.attn_k,
                &blk.attn_v,
                &normed,
                seq_len,
                n_embd,
            )?;

            if let Some(b) = &blk.attn_q_bias {
                for row in q.chunks_exact_mut(b.len()) {
                    for (dst, &bv) in row.iter_mut().zip(b.iter()) {
                        *dst += bv;
                    }
                }
            }
            if let Some(b) = &blk.attn_k_bias {
                for row in k.chunks_exact_mut(b.len()) {
                    for (dst, &bv) in row.iter_mut().zip(b.iter()) {
                        *dst += bv;
                    }
                }
            }
            if let Some(b) = &blk.attn_v_bias {
                for row in v.chunks_exact_mut(b.len()) {
                    for (dst, &bv) in row.iter_mut().zip(b.iter()) {
                        *dst += bv;
                    }
                }
            }

            for s in 0..seq_len {
                let pos = (start_pos + s).min(8191);
                let cos_row = &self.rope_cos[pos * half_dim..(pos + 1) * half_dim];
                let sin_row = &self.rope_sin[pos * half_dim..(pos + 1) * half_dim];
                for h in 0..n_heads {
                    let base = (s * n_heads + h) * head_dim;
                    for j in 0..half_dim {
                        let x0 = q[base + j];
                        let x1 = q[base + half_dim + j];
                        q[base + j] = x0 * cos_row[j] - x1 * sin_row[j];
                        q[base + half_dim + j] = x1 * cos_row[j] + x0 * sin_row[j];
                    }
                }
                for kh in 0..n_kv_heads {
                    let base = (s * n_kv_heads + kh) * head_dim;
                    for j in 0..half_dim {
                        let x0 = k[base + j];
                        let x1 = k[base + half_dim + j];
                        k[base + j] = x0 * cos_row[j] - x1 * sin_row[j];
                        k[base + half_dim + j] = x1 * cos_row[j] + x0 * sin_row[j];
                    }
                }
            }

            blk.k_cache.extend_from_slice(&k);
            blk.v_cache.extend_from_slice(&v);

            let total_seq = blk.k_cache.len() / (n_kv_heads * head_dim);
            let gqa_ratio = n_heads / n_kv_heads;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let k_cache_ref = &blk.k_cache;
            let v_cache_ref = &blk.v_cache;
            let mut attn_out = vec![0.0f32; seq_len * n_heads * head_dim];

            for s in 0..seq_len {
                let valid_t = (start_pos + s + 1).min(total_seq);
                let q_s = &q[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
                let out_s = &mut attn_out[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];

                out_s
                    .par_chunks_mut(head_dim)
                    .enumerate()
                    .for_each(|(h, out_h)| {
                        let kh = h / gqa_ratio;
                        let q_h = &q_s[h * head_dim..(h + 1) * head_dim];
                        let mut scores = vec![0.0f32; valid_t];
                        let mut max_score = f32::NEG_INFINITY;
                        for t in 0..valid_t {
                            let k_t = &k_cache_ref[(t * n_kv_heads + kh) * head_dim
                                ..(t * n_kv_heads + kh + 1) * head_dim];
                            let sc = unsafe { vec_dot_f32_avx2(q_h, k_t) } * scale;
                            scores[t] = sc;
                            if sc > max_score {
                                max_score = sc;
                            }
                        }
                        let mut sum_exp = 0.0f32;
                        for sc in &mut scores {
                            *sc = (*sc - max_score).exp();
                            sum_exp += *sc;
                        }
                        let inv_sum = 1.0f32 / sum_exp.max(1e-12);
                        for t in 0..valid_t {
                            let att = scores[t] * inv_sum;
                            let v_t = &v_cache_ref[(t * n_kv_heads + kh) * head_dim
                                ..(t * n_kv_heads + kh + 1) * head_dim];
                            unsafe {
                                vec_axpy_f32_avx2(out_h, att, v_t);
                            }
                        }
                    });
            }

            let proj = blk
                .attn_output
                .matmul_batched(&attn_out, seq_len, n_heads * head_dim)?;
            for i in 0..x.len() {
                x[i] += proj[i];
            }

            let ffn_in = rms_norm_rows(&x, &blk.ffn_norm, rms_eps);
            let down = MmapWeight::ffn_swiglu_batched(
                &blk.ffn_gate,
                &blk.ffn_up,
                &blk.ffn_down,
                &ffn_in,
                seq_len,
                n_embd,
            )?;
            for i in 0..x.len() {
                x[i] += down[i];
            }
        }

        let last_hidden = &x[(seq_len - 1) * n_embd..seq_len * n_embd];
        let normed_last = rms_norm_rows(last_hidden, &self.output_norm, rms_eps);
        let logits = self.output.matmul_batched(&normed_last, 1, n_embd)?;
        Ok(logits)
    }
}

// ============================================================================
// 8. HOVEDGRENSESNITT FOR LOKAL TEKSTGENERERING (CLI OG WINDOWS-GUI)
// ============================================================================

enum ActiveEngine {
    Qwen35(Qwen35NativeEngine),
    Standard(StandardNativeEngine),
}

impl ActiveEngine {
    pub fn get_hybrid_status(&self) -> HybridStatus {
        match self {
            Self::Qwen35(m) => m.get_hybrid_status(),
            Self::Standard(m) => m.get_hybrid_status(),
        }
    }

    pub fn dynamically_adjust_placement(&mut self, snap: &crate::hardware::ResourceSnapshot) -> Option<String> {
        match self {
            Self::Qwen35(m) => m.dynamically_adjust_placement(snap),
            Self::Standard(m) => m.dynamically_adjust_placement(snap),
        }
    }

    fn clear_state(&mut self) {
        match self {
            Self::Qwen35(m) => m.clear_state(),
            Self::Standard(m) => m.clear_state(),
        }
    }

    fn forward(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        match self {
            Self::Qwen35(m) => m.forward(tokens, start_pos, cancel_flag),
            Self::Standard(m) => m.forward(tokens, start_pos, cancel_flag),
        }
    }
}

pub struct LocalLlm {
    engine: ActiveEngine,
    tokenizer: Tokenizer,
    sampler: SelfMadeSampler,
    pub architecture: String,
    pub model_name: String,
    current_pos: usize,
    pending_token: Option<u32>,
    stopped_at_limit: bool,
    last_disable_thinking: bool,
    governor: std::sync::Arc<crate::hardware::ResourceGovernor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EvalDecision {
    Complete,
    NeedMoreTokens,
    ContinueWith(String),
}

fn looks_cut_off(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let last_ch = trimmed.chars().last().unwrap_or('.');
    !matches!(
        last_ch,
        '.' | '!' | '?' | ')' | ']' | '}' | '"' | '»' | '\'' | '`' | '…'
    )
}

fn has_excessive_repetition(tokens: &[u32]) -> bool {
    let n = tokens.len();
    if n >= 16 {
        let last = tokens[n - 1];
        if tokens[n - 16..].iter().all(|&t| t == last) {
            return true;
        }
    }
    for pat_len in 2..=24 {
        if n >= pat_len * 4 {
            let pat = &tokens[n - pat_len..];
            let rep1 = &tokens[n - pat_len * 2..n - pat_len];
            let rep2 = &tokens[n - pat_len * 3..n - pat_len * 2];
            let rep3 = &tokens[n - pat_len * 4..n - pat_len * 3];
            if pat == rep1 && pat == rep2 && pat == rep3 {
                return true;
            }
        }
    }
    false
}

fn strip_continuation_prefix(s: &str) -> &str {
    let trimmed = s.trim_start();
    let upper = trimmed.to_uppercase();
    for prefix in [
        "FORTSETTELSE:",
        "FORTSETTELSE -",
        "FORTSETTELSE",
        "HER ER FORTSETTELSEN:",
    ] {
        if upper.starts_with(prefix) {
            return trimmed[prefix.len()..].trim_start();
        }
    }
    trimmed
}

fn classify_eval_buffer(buf: &str) -> EvalDecision {
    let trimmed = buf.trim_start();
    if trimmed.is_empty() {
        return EvalDecision::NeedMoreTokens;
    }
    let upper = trimmed.to_uppercase();

    for done_kw in [
        "GODT_NOK",
        "GODT NOK",
        "SVARET ER GODT NOK",
        "SVARET ER KOMPLETT",
        "JA, SVARET ER GODT NOK",
    ] {
        if upper.starts_with(done_kw) {
            return EvalDecision::Complete;
        }
        if done_kw.starts_with(&upper) && trimmed.len() < done_kw.len() {
            return EvalDecision::NeedMoreTokens;
        }
    }

    for cont_kw in ["FORTSETTELSE:", "HER ER FORTSETTELSEN:"] {
        if cont_kw.starts_with(&upper) && trimmed.len() < cont_kw.len() {
            return EvalDecision::NeedMoreTokens;
        }
    }

    if upper.starts_with("FORTSETTELSE:") || upper.starts_with("HER ER FORTSETTELSEN:") {
        let after = strip_continuation_prefix(trimmed);
        if after.is_empty() {
            return EvalDecision::NeedMoreTokens;
        }
        return EvalDecision::ContinueWith(after.to_string());
    }

    if trimmed.chars().count() < 16 {
        return EvalDecision::NeedMoreTokens;
    }

    EvalDecision::ContinueWith(trimmed.to_string())
}

impl LocalLlm {
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        println!(
            "Minnekartlegger GGUF-modell (zero-copy MapViewOfFile) fra {:?}...",
            model_path
        );
        let gguf = GgufModelFile::open(model_path)?;

        let architecture = gguf
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();

        let model_name = gguf
            .metadata
            .get("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                model_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("GGUF-modell")
            })
            .to_string();

        println!(
            "Bygger 100% egenutviklet Rust AVX2+FMA inferensmotor for '{}' (arkitektur: '{}')...",
            model_name, architecture
        );

        let engine = if architecture == "qwen35" {
            ActiveEngine::Qwen35(Qwen35NativeEngine::from_gguf(&gguf)?)
        } else {
            ActiveEngine::Standard(StandardNativeEngine::from_gguf(&gguf, &architecture)?)
        };

        println!("Laster tokenizer fra {:?}...", tokenizer_path);
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Kunne ikke laste tokenizer: {}", e))?;

        let sampler = SelfMadeSampler::new(299792458, 0.7, 0.9);
        let governor = std::sync::Arc::new(crate::hardware::ResourceGovernor::new());

        Ok(Self {
            engine,
            tokenizer,
            sampler,
            architecture,
            model_name,
            current_pos: 0,
            pending_token: None,
            stopped_at_limit: false,
            last_disable_thinking: true,
            governor,
        })
    }

    pub fn get_hybrid_status(&self) -> HybridStatus {
        self.engine.get_hybrid_status()
    }

    pub fn set_resource_governor(
        &mut self,
        governor: std::sync::Arc<crate::hardware::ResourceGovernor>,
    ) {
        self.governor = governor;
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        let mut eos_ids: Vec<u32> = vec![2, 151643, 151645, 128001, 128009, 248044, 248046];
        for tok_str in [
            "<|im_end|>",
            "<|endoftext|>",
            "</s>",
            "<|eot_id|>",
            "<|end_of_text|>",
        ] {
            if let Some(id) = self.tokenizer.token_to_id(tok_str) {
                if !eos_ids.contains(&id) {
                    eos_ids.push(id);
                }
            }
        }
        eos_ids
    }

    /// Prosesserer tokens i dynamiske del-batches tilpasset tilgjengelig RAM og CPU-last.
    /// Dette eliminerer store midlertidige minneallokeringer og hindrer at maskinen fryser.
    fn prefill_tokens_chunked<S>(
        &mut self,
        tokens: &[u32],
        cancel_flag: Option<&AtomicBool>,
        on_status: &mut S,
    ) -> Result<Vec<f32>>
    where
        S: FnMut(&str),
    {
        if tokens.is_empty() {
            bail!("Ingen tokens å prosessere i prefill");
        }
        let mut offset = 0usize;
        let mut last_logits = Vec::new();
        while offset < tokens.len() {
            let snap = self.governor.pace_and_recover(cancel_flag, &mut *on_status)?;
            if let Some(msg) = self.engine.dynamically_adjust_placement(&snap) {
                on_status(&msg);
            }
            let chunk_size = snap.prefill_chunk_size.max(1);
            let end = (offset + chunk_size).min(tokens.len());

            if tokens.len() > 28 && offset > 0 {
                on_status(&format!(
                    "Prosesserer kontekst ({}/{} tokens) – {}...",
                    offset,
                    tokens.len(),
                    snap.status_badge()
                ));
            }

            let slice = &tokens[offset..end];
            let pos = self.current_pos;
            let engine_ref = &mut self.engine;
            last_logits = self
                .governor
                .run_in_optimal_pool(snap.level, || engine_ref.forward(slice, pos, cancel_flag))?;
            self.current_pos += slice.len();
            offset = end;
        }
        Ok(last_logits)
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_with_callback(
            "Du er en ekstremt kortfattet assistent. Svar kun med selve svaret på norsk i maksimalt 1–2 korte setninger uten innledning. Om ikke mulig, øk antall setninger med 1 inntil det gir mening.",
            true,
            prompt,
            max_tokens,
            0.7,
            0.9,
            None,
            |piece| {
                print!("{}", piece);
                let _ = std::io::stdout().flush();
                true
            },
        )
    }

    pub fn generate_with_callback<F>(
        &mut self,
        system_prompt: &str,
        disable_thinking: bool,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        cancel_flag: Option<&AtomicBool>,
        on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str) -> bool,
    {
        self.generate_with_callback_and_status(
            system_prompt,
            disable_thinking,
            prompt,
            max_tokens,
            temperature,
            top_p,
            cancel_flag,
            |_| {},
            on_token,
        )
    }

    pub fn generate_with_callback_and_status<S, F>(
        &mut self,
        system_prompt: &str,
        disable_thinking: bool,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        cancel_flag: Option<&AtomicBool>,
        mut on_status: S,
        mut on_token: F,
    ) -> Result<String>
    where
        S: FnMut(&str),
        F: FnMut(&str) -> bool,
    {
        crate::hardware::lower_current_thread_priority(false);
        self.sampler.temperature = temperature as f32;
        self.sampler.top_p = top_p as f32;
        self.engine.clear_state();
        self.current_pos = 0;
        self.pending_token = None;
        self.stopped_at_limit = false;
        self.last_disable_thinking = disable_thinking;

        let sys_clean = if system_prompt.trim().is_empty() {
            "Du er en ekstremt kortfattet assistent. Svar kun med selve svaret på norsk i maksimalt 1–2 korte setninger uten innledning. Om ikke mulig, øk antall setninger med 1 inntil det gir mening."
        } else {
            system_prompt.trim()
        };

        let formatted_prompt = if disable_thinking {
            format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                sys_clean, prompt
            )
        } else {
            format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                sys_clean, prompt
            )
        };

        let tokens = self
            .tokenizer
            .encode(formatted_prompt, true)
            .map_err(|e| anyhow!("Tokeniseringsfeil: {}", e))?;

        let prompt_tokens = tokens.get_ids().to_vec();
        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        let eos_ids = self.eos_token_ids();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut prev_decoded_len = 0usize;

        // Dynamisk chunked prefill av promptet
        let logits = self.prefill_tokens_chunked(&prompt_tokens, cancel_flag, &mut on_status)?;
        let mut next_token = self.sampler.sample(&logits);

        if eos_ids.contains(&next_token) {
            return Ok(String::new());
        }

        generated_tokens.push(next_token);
        self.pending_token = Some(next_token);

        if let Ok(decoded) = self.tokenizer.decode(&generated_tokens, true) {
            if decoded.len() > prev_decoded_len {
                if !on_token(&decoded[prev_decoded_len..]) {
                    return Ok(decoded);
                }
                prev_decoded_len = decoded.len();
            }
        }

        for _ in 0..max_tokens.saturating_sub(1) {
            let snap = self.governor.pace_and_recover(cancel_flag, &mut on_status)?;
            if let Some(msg) = self.engine.dynamically_adjust_placement(&snap) {
                on_status(&msg);
            }
            if eos_ids.contains(&next_token) {
                self.pending_token = None;
                break;
            }
            let tok_to_feed = self.pending_token.take().unwrap_or(next_token);
            let pos = self.current_pos;
            let engine_ref = &mut self.engine;
            let logits = self.governor.run_in_optimal_pool(snap.level, || {
                engine_ref.forward(&[tok_to_feed], pos, cancel_flag)
            })?;
            self.current_pos += 1;

            next_token = self.sampler.sample(&logits);
            if eos_ids.contains(&next_token) {
                self.pending_token = None;
                break;
            }

            generated_tokens.push(next_token);
            self.pending_token = Some(next_token);

            if has_excessive_repetition(&generated_tokens) {
                break;
            }

            if let Ok(decoded) = self.tokenizer.decode(&generated_tokens, true) {
                if decoded.len() > prev_decoded_len {
                    let piece = &decoded[prev_decoded_len..];
                    if !on_token(piece) {
                        break;
                    }
                    prev_decoded_len = decoded.len();
                }
            }
        }

        if self.pending_token.is_some() {
            self.stopped_at_limit = true;
        }

        println!();
        let final_text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow!("Dekodingsfeil: {}", e))?;
        Ok(final_text)
    }

    /// Gjenbruker eksisterende KV-cache / SSM-tilstand for å raskt fullføre avkuttede setninger
    /// og vurdere om svaret er godt nok uten å måtte prosessere hele samtalen på nytt.
    pub fn evaluate_and_continue_with_callback<S, F>(
        &mut self,
        current_answer: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        cancel_flag: Option<&AtomicBool>,
        mut on_status: S,
        mut on_token: F,
    ) -> Result<String>
    where
        S: FnMut(&str),
        F: FnMut(&str) -> bool,
    {
        crate::hardware::lower_current_thread_priority(false);
        self.sampler.temperature = temperature as f32;
        self.sampler.top_p = top_p as f32;
        let eos_ids = self.eos_token_ids();
        let mut accumulated_answer = current_answer.to_string();

        // Steg 1: Hvis modellen traff maks-tokens midt i en setning, la den først skrive ferdig
        // den påbegynte tanken/setningen direkte fra eksisterende posisjon uten avbrudd.
        if self.stopped_at_limit && looks_cut_off(&accumulated_answer) && self.pending_token.is_some()
        {
            on_status("Fullfører påbegynt setning (gjenbruker hurtigbuffer)...");
            let mut tail_tokens: Vec<u32> = Vec::new();
            let mut prev_len = 0usize;
            let extra_budget = (max_tokens / 2).clamp(80, 260);

            for step in 0..extra_budget {
                let snap = self.governor.pace_and_recover(cancel_flag, &mut on_status)?;
                if let Some(msg) = self.engine.dynamically_adjust_placement(&snap) {
                    on_status(&msg);
                }
                let Some(tok_to_feed) = self.pending_token.take() else {
                    break;
                };
                let pos = self.current_pos;
                let engine_ref = &mut self.engine;
                let logits = self.governor.run_in_optimal_pool(snap.level, || {
                    engine_ref.forward(&[tok_to_feed], pos, cancel_flag)
                })?;
                self.current_pos += 1;

                let next_token = self.sampler.sample(&logits);
                if eos_ids.contains(&next_token) {
                    self.pending_token = None;
                    break;
                }
                tail_tokens.push(next_token);
                self.pending_token = Some(next_token);

                if has_excessive_repetition(&tail_tokens) {
                    break;
                }

                if let Ok(decoded) = self.tokenizer.decode(&tail_tokens, true) {
                    if decoded.len() > prev_len {
                        let piece = &decoded[prev_len..];
                        accumulated_answer.push_str(piece);
                        if !on_token(piece) {
                            return Ok(accumulated_answer);
                        }
                        prev_len = decoded.len();
                    }
                }

                if step >= 24 && !looks_cut_off(&accumulated_answer) {
                    break;
                }
            }
        }

        // Steg 2: Inntil 2 runder med lynrask tilstandsbasert egenvurdering / fortsettelse
        for round in 0..2 {
            if round > 0 && !looks_cut_off(&accumulated_answer) {
                break;
            }

            let is_cut = looks_cut_off(&accumulated_answer);
            on_status(if is_cut {
                "Svaret ble kuttet – fortsetter direkte fra hurtigbuffer..."
            } else {
                "Vurderer om svaret er godt nok..."
            });

            let turn_prompt = if is_cut {
                if self.last_disable_thinking {
                    "<|im_end|>\n<|im_start|>user\nSvaret ovenfor ble avbrutt før det var ferdig. Fortsett direkte der det sluttet, uten innledning og uten å gjenta noe.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                } else {
                    "<|im_end|>\n<|im_start|>user\nSvaret ovenfor ble avbrutt før det var ferdig. Fortsett direkte der det sluttet, uten innledning og uten å gjenta noe.<|im_end|>\n<|im_start|>assistant\n"
                }
            } else if self.last_disable_thinking {
                "<|im_end|>\n<|im_start|>user\nVurder svaret ditt ovenfor. Er spørsmålet godt nok og fullstendig besvart? Hvis det er godt nok, svar kun: GODT_NOK\nHvis noe viktig mangler, skriv kun selve fortsettelsen direkte på norsk uten å gjenta det som allerede er skrevet.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            } else {
                "<|im_end|>\n<|im_start|>user\nVurder svaret ditt ovenfor. Er spørsmålet godt nok og fullstendig besvart? Hvis det er godt nok, svar kun: GODT_NOK\nHvis noe viktig mangler, skriv kun selve fortsettelsen direkte på norsk uten å gjenta det som allerede er skrevet.<|im_end|>\n<|im_start|>assistant\n"
            };

            let encoded = self
                .tokenizer
                .encode(turn_prompt, true)
                .map_err(|e| anyhow!("Tokeniseringsfeil i vurdering: {}", e))?;
            let mut turn_ids = encoded.get_ids().to_vec();
            if let Some(im_end_id) = self.tokenizer.token_to_id("<|im_end|>") {
                if turn_ids.len() >= 2 && turn_ids[0] != im_end_id && turn_ids[1] == im_end_id {
                    turn_ids.remove(0);
                }
            }

            let mut delta_tokens: Vec<u32> = Vec::with_capacity(turn_ids.len() + 1);
            if let Some(unfed) = self.pending_token.take() {
                delta_tokens.push(unfed);
            }
            delta_tokens.extend_from_slice(&turn_ids);

            let logits = self.prefill_tokens_chunked(&delta_tokens, cancel_flag, &mut on_status)?;
            let mut next_token = self.sampler.sample(&logits);
            if eos_ids.contains(&next_token) {
                self.pending_token = None;
                break;
            }

            let mut round_tokens: Vec<u32> = vec![next_token];
            self.pending_token = Some(next_token);
            let mut prev_decoded_len = 0usize;
            let mut eval_buffer = String::new();
            let mut streaming_started = false;
            let mut added_any_text = false;

            let emit_separator = |acc: &mut String, cb: &mut F| -> bool {
                if acc.is_empty() {
                    return true;
                }
                let sep = if looks_cut_off(acc) {
                    if acc.ends_with(char::is_whitespace) {
                        ""
                    } else {
                        " "
                    }
                } else {
                    "\n\n"
                };
                if !sep.is_empty() {
                    acc.push_str(sep);
                    return cb(sep);
                }
                true
            };

            if let Ok(decoded) = self.tokenizer.decode(&round_tokens, true) {
                if decoded.len() > prev_decoded_len {
                    eval_buffer.push_str(&decoded[prev_decoded_len..]);
                    prev_decoded_len = decoded.len();
                }
            }

            let round_budget = ((max_tokens * 3) / 4).max(160);
            for _ in 0..round_budget.saturating_sub(1) {
                if !streaming_started {
                    match classify_eval_buffer(&eval_buffer) {
                        EvalDecision::Complete => {
                            self.pending_token = None;
                            break;
                        }
                        EvalDecision::ContinueWith(cleaned) => {
                            on_status("Utvider og fullfører svaret automatisk...");
                            streaming_started = true;
                            if !emit_separator(&mut accumulated_answer, &mut on_token) {
                                return Ok(accumulated_answer);
                            }
                            accumulated_answer.push_str(&cleaned);
                            added_any_text = true;
                            if !on_token(&cleaned) {
                                return Ok(accumulated_answer);
                            }
                        }
                        EvalDecision::NeedMoreTokens => {}
                    }
                }

                let snap = self.governor.pace_and_recover(cancel_flag, &mut on_status)?;
                if eos_ids.contains(&next_token) {
                    self.pending_token = None;
                    break;
                }

                let tok_to_feed = self.pending_token.take().unwrap_or(next_token);
                let pos = self.current_pos;
                let engine_ref = &mut self.engine;
                let logits = self.governor.run_in_optimal_pool(snap.level, || {
                    engine_ref.forward(&[tok_to_feed], pos, cancel_flag)
                })?;
                self.current_pos += 1;

                next_token = self.sampler.sample(&logits);
                if eos_ids.contains(&next_token) {
                    self.pending_token = None;
                    break;
                }

                round_tokens.push(next_token);
                self.pending_token = Some(next_token);

                if has_excessive_repetition(&round_tokens) {
                    break;
                }

                if let Ok(decoded) = self.tokenizer.decode(&round_tokens, true) {
                    if decoded.len() > prev_decoded_len {
                        let piece = &decoded[prev_decoded_len..];
                        prev_decoded_len = decoded.len();
                        if streaming_started {
                            accumulated_answer.push_str(piece);
                            added_any_text = true;
                            if !on_token(piece) {
                                return Ok(accumulated_answer);
                            }
                        } else {
                            eval_buffer.push_str(piece);
                        }
                    }
                }
            }

            // Hvis runden ble ferdig før eval_buffer nådde terskelen (f.eks. en kort avsluttende setning),
            // sørg for at vi tømmer bufferen til UI hvis det ikke var GODT_NOK!
            if !streaming_started {
                let cleaned = strip_continuation_prefix(&eval_buffer);
                if !cleaned.is_empty() && ! matches!(classify_eval_buffer(&eval_buffer), EvalDecision::Complete) {
                    let upper = cleaned.trim().to_uppercase();
                    if !upper.starts_with("GODT_NOK") && !upper.starts_with("GODT NOK") {
                        if !emit_separator(&mut accumulated_answer, &mut on_token) {
                            return Ok(accumulated_answer);
                        }
                        accumulated_answer.push_str(cleaned);
                        added_any_text = true;
                        let _ = on_token(cleaned);
                    }
                }
            }

            if !added_any_text {
                break;
            }
        }

        Ok(accumulated_answer)
    }
}
