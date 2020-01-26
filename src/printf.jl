import Printf: ini_dec, fix_dec, ini_hex, ini_HEX

if VERSION < v"1.1"
  fix_dec(out, d::BFloat16,               flags::String, width::Int, precision::Int, c::Char) = fix_dec(out, Float32(d),          flags, width, precision, c)
  ini_dec(out, d::BFloat16, ndigits::Int, flags::String, width::Int, precision::Int, c::Char) = ini_dec(out, Float32(d), ndigits, flags, width, precision, c)
  ini_hex(out, d::BFloat16, ndigits::Int, flags::String, width::Int, precision::Int, c::Char) = ini_hex(out, Float32(d), ndigits, flags, width, precision, c)
  ini_HEX(out, d::BFloat16, ndigits::Int, flags::String, width::Int, precision::Int, c::Char) = ini_HEX(out, Float32(d), ndigits, flags, width, precision, c)
  ini_hex(out, d::BFloat16,               flags::String, width::Int, precision::Int, c::Char) = ini_hex(out, Float32(d),          flags, width, precision, c)
  ini_HEX(out, d::BFloat16,               flags::String, width::Int, precision::Int, c::Char) = ini_HEX(out, Float32(d),          flags, width, precision, c)
else
  fix_dec(out, d::BFloat16,               flags::String, width::Int, precision::Int, c::Char, digits) = fix_dec(out, Float32(d),          flags, width, precision, c, digits)
  ini_dec(out, d::BFloat16, ndigits::Int, flags::String, width::Int, precision::Int, c::Char, digits) = ini_dec(out, Float32(d), ndigits, flags, width, precision, c, digits)
  ini_hex(out, d::BFloat16, ndigits::Int, flags::String, width::Int, precision::Int, c::Char, digits) = ini_hex(out, Float32(d), ndigits, flags, width, precision, c, digits)
  ini_HEX(out, d::BFloat16, ndigits::Int, flags::String, width::Int, precision::Int, c::Char, digits) = ini_HEX(out, Float32(d), ndigits, flags, width, precision, c, digits)
  ini_hex(out, d::BFloat16,               flags::String, width::Int, precision::Int, c::Char, digits) = ini_hex(out, Float32(d),          flags, width, precision, c, digits)
  ini_HEX(out, d::BFloat16,               flags::String, width::Int, precision::Int, c::Char, digits) = ini_HEX(out, Float32(d),          flags, width, precision, c, digits)
end
