"""This is a module that interprets the rates, ydots, and Jacobian
through sympy"""

import re
from collections import OrderedDict

import sympy

class SympyRates:

    def __init__(self, ctype="Fortran"):

        self.ctype = ctype

        self.symbol_ludict = OrderedDict() # Symbol lookup dictionary

        if self.ctype == "Fortran":
            self.name_density   = 'state % rho'
            self.name_electron_fraction = 'state % y_e'
        else:
            self.name_density   = 'state.rho'
            self.name_electron_fraction = 'state.y_e'

        # Define these for the particular network
        self.name_rate_data = 'screened_rates'
        self.name_y         = 'Y'
        self.name_ydot      = 'ydot'
        self.name_ydot_nuc  = 'ydot_nuc'
        self.name_jacobian  = 'jac'
        self.name_jacobian_nuc  = 'jac'
        self.symbol_ludict['__dens__'] = self.name_density
        self.symbol_ludict['__y_e__'] = self.name_electron_fraction


        self.float_explicit_num_digits = 17

    def ydot_term_symbol(self, rate, y_i):
        """
        return a sympy expression containing this rate's contribution to
        the ydot term for nuclide y_i.
        """
        srate = self.specific_rate_symbol(rate)

        # Check if y_i is a reactant or product
        c_reac = rate.reactants.count(y_i)
        c_prod = rate.products.count(y_i)
        if c_reac == 0 and c_prod == 0:
            # The rate doesn't contribute to the ydot for this y_i
            ydot_sym = float(sympy.sympify(0.0))
        else:
            # y_i appears as a product or reactant
            ydot_sym = (c_prod - c_reac) * srate
        return ydot_sym.evalf(n=self.float_explicit_num_digits)

    def specific_rate_symbol(self, rate):
        """
        return a sympy expression containing the term in a dY/dt equation
        in a reaction network corresponding to this rate.

        Also enter the symbol and substitution in the lookup table.
        """

        # composition dependence
        Y_sym = 1
        for r in sorted(set(rate.reactants)):
            print(r)
            c = rate.reactants.count(r)
            print(c)
            if self.ctype == "Fortran":
                sym_final = f'{self.name_y}(j{r})'
            else:
                sym_final = f'{self.name_y}({r.c()})'

            print(sym_final)
            sym_temp  = f'Y__j{r}__'
            print(sym_temp)

            self.symbol_ludict[sym_temp] = sym_final
            Y_sym = Y_sym * sympy.symbols(sym_temp)**c
            print(Y_sym)
            print('')

        # density dependence
        dens_sym = sympy.symbols('__dens__')**rate.dens_exp

        # electron fraction if electron capture reaction
        if (rate.weak_type == 'electron_capture' and not rate.tabular):
            y_e_sym = sympy.symbols('__y_e__')
        else:
            y_e_sym = sympy.sympify(1)

        # prefactor
        prefactor_sym = sympy.sympify(1)/sympy.sympify(rate.inv_prefactor)

        # screened rate
        sym_final = self.name_rate_data + f'(k_{rate.fname})'
        sym_temp  = f'NRD__k_{rate.fname}__'
        self.symbol_ludict[sym_temp] = sym_final
        screened_rate_sym = sympy.symbols(sym_temp)

        srate_sym = prefactor_sym * dens_sym * y_e_sym * Y_sym * screened_rate_sym
        return srate_sym

    def jacobian_term_symbol(self, rate, ydot_j, y_i):
        """
        return a sympy expression containing the term in a jacobian matrix
        in a reaction network corresponding to this rate

        Returns the derivative of the j-th YDOT wrt. the i-th Y
        If the derivative is zero, returns 0.

        ydot_j and y_i are objects of the class 'Nucleus'
        """
        ydot_sym = self.ydot_term_symbol(rate, ydot_j)
        deriv_sym = sympy.symbols(f'Y__j{y_i}__')
        jac_sym = sympy.diff(ydot_sym, deriv_sym)
        symbol_is_null = False
        if jac_sym.equals(0):
            symbol_is_null = True
        return (jac_sym.evalf(n=self.float_explicit_num_digits), symbol_is_null)

    def fortranify(self, s):
        """
        Given string s, will replace the symbols appearing as keys in
        self.symbol_ludict with their corresponding entries.
        """
        for k in self.symbol_ludict:
            v = self.symbol_ludict[k]
            s = s.replace(k,v)
        if s == '0':
            s = '0.0e0_rt'

        ## Replace all double precision literals with custom real type
        ## literals
        # constant type specifier
        const_spec = "_rt"

        # we want to replace any "d" scientific notation with the new
        # style this matches stuff like -1.25d-10, and gives us
        # separate groups for the prefix and exponent.  The [^\w]
        # makes sure a letter isn't right in front of the match (like
        # 'k3d-1'). Alternately, we allow for a match at the start of
        # the string.
        d_re = re.compile(r"([^\w\+\-]|\A)([\+\-0-9.][0-9.]+)[dD]([\+\-]?[0-9]+)", re.IGNORECASE|re.DOTALL)

        # update "d" scientific notation -- allow for multiple
        # constants in a single string
        for dd in d_re.finditer(s):
            prefix = dd.group(2)
            exponent = dd.group(3)
            new_num = f"{prefix}e{exponent}{const_spec}"
            old_num = dd.group(0).strip()
            s = s.replace(old_num, new_num)

        return s

    def cxxify(self, s):
        """
        Given string s, will replace the symbols appearing as keys in
        self.symbol_ludict with their corresponding entries.
        """
        for k in self.symbol_ludict:
            v = self.symbol_ludict[k]
            s = s.replace(k,v)
        if s == '0':
            s = '0.0e0'

        ## Replace all double precision literals with custom real type
        ## literals
        # constant type specifier
        const_spec = "_rt"

        # we want append any "e" scientific notation with "_rt".  This
        # matches stuff like -1.25d-10, and gives us separate groups
        # for the prefix and exponent.  The [^\w] makes sure a letter
        # isn't right in front of the match (like
        # 'k3d-1'). Alternately, we allow for a match at the start of
        # the string.
        e_re = re.compile(r"([^\w\+\-]|\A)([\+\-0-9.][0-9.]+)[eE]([\+\-]?[0-9]+)", re.IGNORECASE|re.DOTALL)

        # update "d" scientific notation -- allow for multiple
        # constants in a single string
        for ee in e_re.finditer(s):
            old_num = ee.group(0).strip()
            s = s.replace(old_num, f"{old_num}{const_spec}")

        return s

