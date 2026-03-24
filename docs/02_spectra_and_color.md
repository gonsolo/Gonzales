# Spectra and Color

Physically based rendering requires an accurate model of light color. In the
real world, light is a continuous spectrum of wavelengths. Gonzales currently
uses an RGB approximation — three floating-point values representing red, green,
and blue — but the code is structured to support full spectral rendering in
the future through the `Spectrum` protocol.

## The Spectrum Protocol

All spectral types in gonzales conform to the `Spectrum` protocol, which
requires multiplication and conversion to RGB. This abstraction allows
`PiecewiseLinearSpectrum` (used for metal optical constants) to coexist with
the primary `RgbSpectrum` type, and will later enable a `SampledSpectrum`
for wavelength-by-wavelength rendering.

## RgbSpectrum

The workhorse type is `RgbSpectrum`. Like the geometry types, it is backed
by `SIMD4<Real>` to get vectorized arithmetic for free:

{{snippet:Sources/libgonzales/Core/Spectrum.swift:rgb-spectrum-struct}}

This means that multiplying two spectra — which happens at every surface
interaction — compiles down to a single SIMD multiply instruction. The named
accessors `red`, `green`, `blue` provide readability while the underlying
SIMD4 provides performance.

Global constants `black`, `white`, `gray`, `red`, `green`, `blue` are defined
for convenience and appear throughout the integrator code as sentinel values.

## Metal Optical Constants

Metals like silver, aluminium, copper, brass, and gold have wavelength-dependent
refractive indices and extinction coefficients. Gonzales stores these as
`PiecewiseLinearSpectrum` values — arrays of (wavelength, value) pairs sampled
from measured data. The `namedSpectra` dictionary maps PBRT material names
(e.g. `"metal-Ag-eta"`) to the corresponding spectrum.

## Black-Body Radiation

Light sources like incandescent bulbs and stars emit radiation whose color
depends on temperature. The `blackBodyToRgb` function approximates this using
Tanner Helland's algorithm, avoiding the need for a full spectral integration:

{{snippet:Sources/libgonzales/Core/Spectrum.swift:black-body}}

The function maps temperatures from candlelight (~1800K, warm orange) through
daylight (~6500K, neutral white) to overcast sky (~10000K, bluish white).

## Gamma Correction

Two functions handle the nonlinear sRGB encoding and decoding:
`gammaLinearToSrgb` converts from the linear light space used during rendering
to the sRGB curve expected by displays, while `gammaSrgbToLinear` does the
reverse for texture inputs. Getting this wrong is one of the most common
sources of washed-out or overly dark renders.
