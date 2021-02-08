include(scipp-util)

scipp_unary(math abs)
scipp_unary(math exp)
scipp_unary(math log)
scipp_unary(math log10)
scipp_unary(math reciprocal)
scipp_unary(math sqrt)
setup_scipp_category(math)

scipp_unary(util values SKIP_VARIABLE NO_OUT)
scipp_unary(util variances SKIP_VARIABLE NO_OUT)
setup_scipp_category(util)

scipp_unary(trigonometry sin SKIP_VARIABLE)
scipp_unary(trigonometry cos SKIP_VARIABLE)
scipp_unary(trigonometry tan SKIP_VARIABLE)
scipp_unary(trigonometry asin SKIP_VARIABLE)
scipp_unary(trigonometry acos SKIP_VARIABLE)
scipp_unary(trigonometry atan SKIP_VARIABLE)
setup_scipp_category(trigonometry)

scipp_unary(special_values isnan NO_OUT)
scipp_unary(special_values isinf NO_OUT)
scipp_unary(special_values isfinite NO_OUT SKIP_VARIABLE)
scipp_unary(special_values isposinf NO_OUT)
scipp_unary(special_values isneginf NO_OUT)
setup_scipp_category(special_values)

scipp_binary(comparison equal)
scipp_binary(comparison greater)
scipp_binary(comparison greater_equal)
scipp_binary(comparison less)
scipp_binary(comparison less_equal)
scipp_binary(comparison not_equal)
setup_scipp_category(comparison)

scipp_function("binary" arithmetic operator+ OP plus)
scipp_function("binary" arithmetic operator- OP minus)
scipp_function("binary" arithmetic operator* OP times)
scipp_function("binary" arithmetic operator/ OP divide)
scipp_function("binary" arithmetic operator% OP mod)
scipp_function("inplace" arithmetic operator+= OP plus_equals)
scipp_function("inplace" arithmetic operator-= OP minus_equals)
scipp_function("inplace" arithmetic operator*= OP times_equals)
scipp_function("inplace" arithmetic operator/= OP divide_equals)
scipp_function("inplace" arithmetic operator%= OP mod_equals)
setup_scipp_category(arithmetic)

scipp_function("binary" logical operator| OP logical_or)
scipp_function("binary" logical operator& OP logical_and)
scipp_function("binary" logical operator^ OP logical_xor)
scipp_function("inplace" logical operator|= OP logical_or_equals)
scipp_function("inplace" logical operator&= OP logical_and_equals)
scipp_function("inplace" logical operator^= OP logical_xor_equals)
setup_scipp_category(logical)
