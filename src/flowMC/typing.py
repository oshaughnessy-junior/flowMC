"""Convenience type aliases for jaxtyping scalar annotations."""

from typing import TypeAlias

from jaxtyping import Array, ArrayLike, Complex, Float, Int, Real

FloatScalar: TypeAlias = Float[Array, ""]
IntScalar: TypeAlias = Int[Array, ""]
ComplexScalar: TypeAlias = Complex[Array, ""]
FloatLike: TypeAlias = float | Float[Array, ""]
RealScalarLike: TypeAlias = Real[ArrayLike, ""]
