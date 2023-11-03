// RUN: XDSL_ROUNDTRIP

// CHECK-LABEL: hw.module @test1
// CHECK-SAME: (in %arg0: i3, in %arg1: i1, in %arg2: !hw.array<1000xi8>, out result: i50)
hw.module @test1(in %arg0: i3, in %arg1: i1, in %arg2: !hw.array<1000xi8>, out result: i50) {
  %a = hw.constant 42 : i12

  %small1 = hw.constant 0 : i19
  %small2 = hw.constant 0 : i19
  %idx = hw.constant 0 : i10
  %true = hw.constant true
  %mux = hw.constant 0 : i7

  // CHECK: %s0 = hw.struct_create (%small1, %mux) : !hw.struct<foo: i19, bar: i7>
  %s0 = hw.struct_create (%small1, %mux) : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT: %foo = hw.struct_extract %s0["foo"] : !hw.struct<foo: i19, bar: i7>
  %foo = hw.struct_extract %s0["foo"] : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT: = hw.struct_inject %s0["foo"], {{.*}} : !hw.struct<foo: i19, bar: i7>
  %s1 = hw.struct_inject %s0["foo"], %foo : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT:  %foo_2, %bar = hw.struct_explode %s0 : !hw.struct<foo: i19, bar: i7>
  %foo_2, %bar = hw.struct_explode %s0 : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT: hw.bitcast %s0 : (!hw.struct<foo: i19, bar: i7>)
  %structBits = hw.bitcast %s0 : (!hw.struct<foo: i19, bar: i7>) -> i26

  // CHECK-NEXT: = hw.array_slice %arg2[%idx] : (!hw.array<1000xi8>) -> !hw.array<24xi8>
  %subArray = hw.array_slice %arg2[%idx] : (!hw.array<1000xi8>) -> !hw.array<24xi8>
  // CHECK-NEXT: %arrCreated = hw.array_create %small1, %small2 : i19
  %arrCreated = hw.array_create %small1, %small2 : i19
  // CHECK-NEXT: %arr2 = hw.array_create %small1, %small2, %small1 : i19
  %arr2 = hw.array_create %small1, %small2, %small1 : i19
  // CHECK-NEXT: = hw.array_concat %arrCreated, %arr2 : !hw.array<2xi19>, !hw.array<3xi19>
  %bigArray = hw.array_concat %arrCreated, %arr2 : !hw.array<2 x i19>, !hw.array<3 x i19>
  // CHECK-NEXT: %el = hw.array_get %arrCreated[%true] : !hw.array<2xi19>, i1
  %el = hw.array_get %arrCreated[%true] : !hw.array<2xi19>, i1

  // CHECK-NEXT: hw.aggregate_constant [false, true] : !hw.struct<a: i1, b: i1>
  hw.aggregate_constant [false, true] : !hw.struct<a: i1, b: i1>
  // CHECK-NEXT: hw.aggregate_constant [0 : i2, 1 : i2, -2 : i2, -1 : i2] : !hw.array<4xi2>
  hw.aggregate_constant [0 : i2, 1 : i2, -2 : i2, -1 : i2] : !hw.array<4xi2>
  // CHECK-NEXT: hw.aggregate_constant [false] : !hw.uarray<1xi1>
  hw.aggregate_constant [false] : !hw.uarray<1xi1>
  // CHECK-NEXT: hw.aggregate_constant [[false]] : !hw.struct<a: !hw.array<1xi1>>
  hw.aggregate_constant [[false]] : !hw.struct<a: !hw.array<1xi1>>
  // CHECK-NEXT: hw.aggregate_constant ["A"] : !hw.struct<a: !hw.enum<A, B, C>>
  hw.aggregate_constant ["A"] : !hw.struct<a: !hw.enum<A, B, C>>
  // CHECK-NEXT: hw.aggregate_constant ["A"] : !hw.array<1x!hw.enum<A, B, C>>
  hw.aggregate_constant ["A"] : !hw.array<1 x!hw.enum<A, B, C>>

  // CHECK-NEXT:    hw.output %foo : i19
  hw.output %foo : i19
}
// CHECK-NEXT:  }

// Check that we pass the verifier that the module's function type matches
// the block argument types when using InOutTypes.
// CHECK: hw.module @InOutPort(inout %arg0_1: i1)
hw.module @InOutPort(inout %arg0: i1) { }

// // CHECK-LABEL: hw.module @argRenames
// // CHECK-SAME: attributes {argNames = [""]}
hw.module @argRenames(in %arg1: i32) attributes {argNames = [""]} { }

// // CHECK-LABEL: hw.module @commentModule
// // CHECK-SAME: attributes {comment = "hello world"}
hw.module @commentModule() attributes {comment = "hello world"} { }
