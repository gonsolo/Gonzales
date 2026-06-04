# Spectra and Color

Physically based rendering requires an accurate model of light color. In the
real world, light is a continuous spectrum of wavelengths. Gonzales currently
uses an RGB approximation — three `Float32` values representing red, green,
and blue.

## The RGB Type

The workhorse type is `RGB` in `geometry.mojo`:

```mojo
struct RGB(TrivialRegisterPassable):
    var r: Float32
    var g: Float32
    var b: Float32

    fn luma(self) -> Float32:
        return Float32(0.2126)*self.r + Float32(0.7152)*self.g + Float32(0.0722)*self.b
```

Multiplying two spectra — which happens at every surface interaction to
apply throughput — is three scalar multiplies. `TrivialRegisterPassable`
ensures these land in registers and in GPU-friendly flat path buffers
with no indirection.

Global sentinel values (`RGB(0,0,0)` for black, `RGB(1,1,1)` for white) appear
throughout the shading code.

## Metal Optical Constants

Metals like silver, aluminium, copper, and gold have wavelength-dependent
refractive indices and extinction coefficients. These are stored as arrays of
(wavelength, value) pairs sampled from measured data and looked up at render
time in `shading.mojo`.

## Black-Body Radiation

Light sources like incandescent bulbs emit radiation whose color depends on
temperature. The renderer approximates black-body color using a polynomial fit
that maps temperatures from candlelight (~1800 K, warm orange) through daylight
(~6500 K, neutral white) to overcast sky (~10000 K, bluish white).

## Gamma Correction

Linear-to-sRGB and sRGB-to-linear conversions are applied at texture load time
and image output time respectively. Getting this wrong is one of the most common
sources of washed-out or overly dark renders — all rendering happens in linear
light, displays expect sRGB.
