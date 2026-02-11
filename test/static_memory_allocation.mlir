// RUN: eaac-opt %s -eaac-static-memory-allocation | FileCheck %s

// Test basic static memory allocation
module {
  func.func @test_simple_allocation() {
    // CHECK: memref.alloc() {eaac.static_address = 0 : i64, eaac.static_size = 1024 : i64}
    %buf1 = memref.alloc() : memref<16x16xf32>
    
    // CHECK: memref.alloc() {eaac.static_address = 1024 : i64, eaac.static_size = 2048 : i64}
    %buf2 = memref.alloc() : memref<32x16xf32>
    
    memref.dealloc %buf1 : memref<16x16xf32>
    memref.dealloc %buf2 : memref<32x16xf32>
    return
  }
  
  // Test memory reuse after deallocation
  func.func @test_memory_reuse() {
    // First allocation
    %buf1 = memref.alloc() : memref<64xf32>
    memref.dealloc %buf1 : memref<64xf32>
    
    // Second allocation can reuse the same memory
    %buf2 = memref.alloc() : memref<64xf32>
    memref.dealloc %buf2 : memref<64xf32>
    return
  }
  
  // Test overlapping lifetimes
  func.func @test_overlapping_lifetimes() {
    %buf1 = memref.alloc() : memref<100xf32>
    %buf2 = memref.alloc() : memref<200xf32>
    %buf3 = memref.alloc() : memref<50xf32>
    
    // All three buffers are live at this point, so they need separate addresses
    
    memref.dealloc %buf1 : memref<100xf32>
    memref.dealloc %buf2 : memref<200xf32>
    memref.dealloc %buf3 : memref<50xf32>
    return
  }
  
  // Test different data types
  func.func @test_data_types() {
    %i8_buf = memref.alloc() : memref<128xi8>
    %i16_buf = memref.alloc() : memref<64xi16>
    %i32_buf = memref.alloc() : memref<32xi32>
    %f16_buf = memref.alloc() : memref<64xf16>
    %f32_buf = memref.alloc() : memref<32xf32>
    %f64_buf = memref.alloc() : memref<16xf64>
    
    memref.dealloc %i8_buf : memref<128xi8>
    memref.dealloc %i16_buf : memref<64xi16>
    memref.dealloc %i32_buf : memref<32xi32>
    memref.dealloc %f16_buf : memref<64xf16>
    memref.dealloc %f32_buf : memref<32xf32>
    memref.dealloc %f64_buf : memref<16xf64>
    return
  }
}
