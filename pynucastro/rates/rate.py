"""
Classes and methods to interface with files storing rate data.
"""

import os
import re
import io
import numpy as np
import matplotlib.pyplot as plt
import numba
#from sympy import *
#from sympy.abc import x
from collections import Counter

try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass

from pynucastro.nucdata import UnidentifiedElement, PeriodicTable, PartitionFunctionCollection, BindingTable, SpinTable

import sys
sys.path.append(os.path.abspath("/scratch/jh2/ps3459/pynucastro/pynucastro/rates"))
import constants as cons


#print(constants())

_pynucastro_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_pynucastro_rates_dir = os.path.join(_pynucastro_dir, 'library')
_pynucastro_tabular_dir = os.path.join(_pynucastro_rates_dir, 'tabular')


#read the spin table once and store it at the module-level
_spin_table = SpinTable(set_double_gs=False)

# read the binding energy table once and store it at the module-level
_binding_table = BindingTable()


_pcollection = PartitionFunctionCollection(use_high_temperatures = True, use_set='frdm')

def _find_rate_file(ratename):
    """locate the Reaclib or tabular rate or library file given its name.  Return
    None if the file cannot be located, otherwise return its path."""

    # check to see if the rate file is in the working dir or
    # is already the full path
    x = ratename
    if os.path.isfile(x):
        return os.path.realpath(x)

    # check to see if the rate file is in pynucastro/library
    x = os.path.join(_pynucastro_rates_dir, ratename)
    if os.path.isfile(x):
        return os.path.realpath(x)

    # check to see if the rate file is in pynucastro/library/tabular
    x = os.path.join(_pynucastro_tabular_dir, ratename)
    if os.path.isfile(x):
        return os.path.realpath(x)

    # notify user we can't find the file
    raise Exception(f'File {ratename} not found in the working directory, {_pynucastro_rates_dir}, or {_pynucastro_tabular_dir}')



Tfactor_spec = [
('T9', numba.float64),
('T9i', numba.float64),
('T913', numba.float64),
('T913i', numba.float64),
('T953', numba.float64),
('lnT9', numba.float64)
]

@jitclass(Tfactor_spec)
class Tfactors:
    """ precompute temperature factors for speed """

    def __init__(self, T):
        """ return the Tfactors object.  Here, T is temperature in Kelvin """
        self.T9 = T/1.e9
        self.T9i = 1.0/self.T9
        self.T913i = self.T9i**(1./3.)
        self.T913 = self.T9**(1./3.)
        self.T953 = self.T9**(5./3.)
        self.lnT9 = np.log(self.T9)


class SingleSet:
    """ a set in Reaclib is one piece of a rate, in the form

        lambda = exp[ a_0 + sum_{i=1}^5  a_i T_9**(2i-5)/3  + a_6 log T_9]

        A single rate in Reaclib can be composed of multiple sets
    """

    def __init__(self, a, labelprops=None):
        """here a is iterable (e.g., list or numpy array), storing the
           coefficients, a0, ..., a6

        """
        self.a = a
        self.labelprops = labelprops
        self.label = None
        self.resonant = None
        self.weak = None
        self.reverse = None

        self._update_label_properties()

    def _update_label_properties(self):
        """ Set label and flags indicating Set is resonant,
            weak, or reverse. """
        assert isinstance(self.labelprops, str)
        try:
            assert len(self.labelprops) == 6
        except:
            raise
        else:
            self.label = self.labelprops[0:4]
            self.resonant = self.labelprops[4] == 'r'
            self.weak = self.labelprops[4] == 'w'
            self.reverse = self.labelprops[5] == 'v'

    def __eq__(self, other):
        """ Determine whether two SingleSet objects are equal to each other. """
        x = True

        for ai, aj in zip(self.a, other.a):
            x = x and (ai == aj)

        x = x and (self.label == other.label)
        x = x and (self.resonant == other.resonant)
        x = x and (self.weak == other.weak)
        x = x and (self.reverse == other.reverse)
        return x

    def f(self):
        """
        return a function for this set -- note: Tf here is a Tfactors
        object
        """
        return lambda tf: np.exp(self.a[0] +
                                 self.a[1]*tf.T9i +
                                 self.a[2]*tf.T913i +
                                 self.a[3]*tf.T913 +
                                 self.a[4]*tf.T9 +
                                 self.a[5]*tf.T953 +
                                 self.a[6]*tf.lnT9)

    def set_string(self, prefix="set", plus_equal=False):
        """
        return a string containing the python code for this set
        """
        if plus_equal:
            string = f"{prefix} += np.exp( "
        else:
            string = f"{prefix} = np.exp( "
        string += f" {self.a[0]}"
        if not self.a[1] == 0.0:
            string += f" + {self.a[1]}*tf.T9i"
        if not self.a[2] == 0.0:
            string += f" + {self.a[2]}*tf.T913i"
        if not self.a[3] == 0.0:
            string += f" + {self.a[3]}*tf.T913"
        if not (self.a[4] == 0.0 and self.a[5] == 0.0 and self.a[6] == 0.0):
            string += "\n{}         ".format(len(prefix)*" ")
        if not self.a[4] == 0.0:
            string += f" + {self.a[4]}*tf.T9"
        if not self.a[5] == 0.0:
            string += f" + {self.a[5]}*tf.T953"
        if not self.a[6] == 0.0:
            string += f" + {self.a[6]}*tf.lnT9"
        string += ")"
        return string


class UnsupportedNucleus(BaseException):
    def __init__(self):
        return


class Nucleus:
    """
    a nucleus that participates in a reaction -- we store it in a
    class to hold its properties, define a sorting, and give it a
    pretty printing string.

    :var Z:               atomic number
    :var N:               neutron number
    :var A:               atomic mass
    :var nucbind:         nuclear binding energy (MeV / nucleon)
    :var short_spec_name: nucleus abbrevation (e.g. "he4")
    :var caps_name:       capitalized short species name (e.g. "He4")
    :var el:              element name (e.g. "he")
    :var pretty:          LaTeX formatted version of the nucleus name

    """
    def __init__(self, name, dummy=False):
        name = name.lower()
        self.raw = name

        # a dummy nucleus is one that we can use where a nucleus is needed
        # but it is not considered to be part of the network
        self.dummy = dummy

        # element symbol and atomic weight
        if name == "p":
            self.el = "h"
            self.A = 1
            self.short_spec_name = "h1"
            self.caps_name = "H1"
        elif name == "d":
            self.el = "h"
            self.A = 2
            self.short_spec_name = "h2"
            self.caps_name = "H2"
        elif name == "t":
            self.el = "h"
            self.A = 3
            self.short_spec_name = "h3"
            self.caps_name = "H3"
        elif name == "a":
            #this is a convenience, enabling the use of a commonly-used alias:
            #    He4 --> \alpha --> "a" , e.g. c12(a,g)o16
            self.el ="he"
            self.A = 4
            self.short_spec_name = "he4"
            self.raw = "he4"
            self.caps_name = "He4"
        elif name == "n":
            self.el = "n"
            self.A = 1
            self.Z = 0
            self.N = 1
            self.short_spec_name = "n"
            self.spec_name = "neutron"
            self.pretty = fr"\mathrm{{{self.el}}}"
            self.caps_name = "N"
        else:
            e = re.match(r"([a-zA-Z]*)(\d*)", name)
            self.el = e.group(1).title()  # chemical symbol
            assert self.el
            try:
                self.A = int(e.group(2))
            except:
                if (name.strip() == 'al-6' or
                    name.strip() == 'al*6'):
                    raise UnsupportedNucleus()
                else:
                    raise
            assert self.A >= 0
            self.short_spec_name = name
            self.caps_name = name.capitalize()

        # set the number of spin states
        try:
            self.spin_states = _spin_table.get_spin_nuclide(self.short_spec_name).spin_states
        except NotImplementedError:
            self.spin_states = None

        # use lowercase element abbreviation regardless the case of the input
        self.el = self.el.lower()

        # set a partition function object to every nucleus
        self.partition_function = _pcollection.get_partition_function(self.short_spec_name)

        # atomic number comes from periodic table
        if name != "n":
            try:
                i = PeriodicTable.lookup_abbreviation(self.el)
            except UnidentifiedElement:
                print(f'Could not identify element: {self.el}')
                raise
            except:
                raise
            else:
                self.Z = i.Z
                assert isinstance(self.Z, int)
                assert self.Z >= 0
                self.N = self.A - self.Z
                assert isinstance(self.N, int)
                assert self.N >= 0

                # long name
                self.spec_name = f'{i.name}-{self.A}'

                # latex formatted style
                self.pretty = fr"{{}}^{{{self.A}}}\mathrm{{{self.el.capitalize()}}}"

        try:
            self.nucbind = _binding_table.get_nuclide(n=self.N, z=self.Z).nucbind
        except NotImplementedError:
            # the binding energy table doesn't know about this nucleus
            self.nucbind = None

    def __repr__(self):
        return self.raw

    def __hash__(self):
        return hash((self.Z, self.A))

    def c(self):
        """return the name capitalized"""
        return self.caps_name

    def __eq__(self, other):
        if isinstance(other, Nucleus):
            return self.el == other.el and \
               self.Z == other.Z and self.A == other.A
        if isinstance(other, tuple):
            return (self.Z, self.A) == other
        return NotImplemented

    def __lt__(self, other):
        if not self.Z == other.Z:
            return self.Z < other.Z
        return self.A < other.A

class UnsupportedChemSpecie(BaseException):
    def __init__(self):
        print('The chemical specie you entered is unsupported')
        return

class ChemSpecie:

    """
    a class like the nucleus class above but for chemical species in the ISM
    :var Z:               atomic number
    :var m:               total mass of specie in g
    :var N:               neutron number
    :var e:               number of electrons
    :var gamma:           adiabatic index
    :var chemsign:            chemical sign

    """
    def __init__(self, name, dummy=False):

        #importing sympy here because it interferes with the rest of pynucastro stuff if imported at the top
        import sympy as sp

        # a dummy chemical specie is one that we can use where a chemical specie is needed
        # but it is not considered to be part of the network
        self.dummy = dummy
        self.raw = name

        self.num = 0
        self.end = 1
        # element symbol and atomic weight
        if name.casefold() == "elec" or name.casefold() == 'e':
            self.Z = 0 #number of protons
            self.m = 9.10938188e-28 #mass in g
            self.N = 0 #number of neutrons
            self.e = 1 #number of electons
            self.gamma = 1.66666667 #adiabatic index
            self.chemsign = "E"
            self.A = self.Z + self.N
        elif name.casefold() == "hp" or name.casefold() == "h+":
            self.Z = 1
            self.m = 1.67262158e-24
            self.N = 0
            self.e = 0
            self.gamma = 1.66666667
            self.chemsign = "H+"
            self.A = self.Z + self.N
        elif name.casefold() == "h":
            self.Z = 1
            self.m = 1.67353251819e-24
            self.N = 0
            self.e = 1
            self.gamma = 1.66666667
            self.chemsign = "H"
            self.A = self.Z + self.N
        elif name.casefold() == "hm" or name.casefold() == 'h-':
            self.Z = 1
            self.m = 1.67444345638e-24
            self.N = 0
            self.e = 2
            self.gamma = 1.66666667
            self.chemsign = "H-"
            self.A = self.Z + self.N
        elif name.casefold() == "dp" or name.casefold() == "d+":
            self.Z = 1
            self.m = 3.34512158e-24
            self.N = 1
            self.e = 0
            self.gamma = 1.66666667
            self.chemsign = "D+"
            self.A = self.Z + self.N
        elif name.casefold() == "d":
            self.Z = 1
            self.m = 3.34603251819e-24
            self.N = 1
            self.e = 1
            self.gamma = 1.66666667
            self.chemsign = "D"
            self.A = self.Z + self.N
        elif name.casefold() == "h2p" or name.casefold() == "h2+":
            self.Z = 2
            self.m = 3.34615409819e-24
            self.N = 0
            self.e = 1
            self.gamma = 1.4
            self.chemsign = "H2+"
            self.A = self.Z + self.N
        elif name.casefold() == "dm" or name.casefold() == "d-":
            self.Z = 1
            self.m = 3.34694345638e-24
            self.N = 1
            self.e = 2
            self.gamma = 1.66666667
            self.chemsign = "D-"
            self.A = self.Z + self.N
        elif name.casefold() == "h2":
            self.Z = 2
            self.m = 3.34706503638e-24
            self.N = 0
            self.e = 2
            self.gamma = 1.4
            self.chemsign = "H2"
            self.A = self.Z + self.N
        elif name.casefold() == "hdp" or name.casefold() == "hd+":
            self.Z = 2
            self.m = 5.01865409819e-24
            self.N = 1
            self.e = 1
            self.gamma = 1.4
            self.chemsign = "HD+"
            self.A = self.Z + self.N
        elif name.casefold() == "hd":
            self.Z = 2
            self.m = 5.01956503638e-24
            self.N = 1
            self.e = 2
            self.gamma = 1.4
            self.chemsign = "HD"
            self.A = self.Z + self.N
        elif name.casefold() == "hepp" or name.casefold() == "he++":
            self.Z = 2
            self.m = 6.69024316e-24
            self.N = 2
            self.e = 0
            self.gamma = 1.66666667
            self.chemsign = "HE++"
            self.A = self.Z + self.N
        elif name.casefold() == "hep" or name.casefold() == "he+":
            self.Z = 2
            self.m = 6.69115409819e-24
            self.N = 2
            self.e = 1
            self.gamma = 1.66666667
            self.chemsign = "HE+"
            self.A = self.Z + self.N
        elif name.casefold() == "he":
            self.Z = 2
            self.m = 6.69206503638e-24
            self.N = 2
            self.e = 2
            self.gamma = 1.66666667
            self.chemsign = "He"
            self.A = self.Z + self.N
        elif name.casefold() == "dummy":
            self.Z = 0
            self.m = 0
            self.N = 0
            self.e = 0
            self.gamma = 0
            self.chemsign = "dummy"
            self.A = self.Z + self.N
        else:
            raise UnsupportedChemSpecie()

        self.sym_name = sp.symbols(self.chemsign, real=True)


    def __iter__(self):
        #print('iterrr')
        #yield 
        #yield from {
         #   "Z": self.Z,
         #   "m": self.m,
         #   "N": self.N,
         #   "e": self.e,
         #   "gamma": self.gamma,
         #   "chemsign": self.chemsign
        #}.items()
        return self

    def __next__(self):
        if self.num > self.end:
            raise StopIteration
        else:
            self.num += 1
            return self.num - 1


    def __repr__(self):
        return self.chemsign

    def __hash__(self):
        return hash((self.Z, self.m, self.N, self.e, self.gamma, self.chemsign, self.A, self.sym_name))

    def __eq__(self, other):
        #if isinstance(other, ChemSpecie):
        #    return self.Z == other.Z and self.m == other.m and \
        #       self.N == other.N and self.e == other.e and \
        #       self.gamma == other.gamma and self.chemsign == other.chemsign
        #if isinstance(other, tuple):
        #    return (self.Z, self.A) == other
        #return NotImplemented
        return (self.Z, self.m, self.N, self.e, self.gamma, self.chemsign, self.A, self.sym_name) == \
               (other.Z, other.m, other.N, other.e, other.gamma, other.chemsign, other.A, other.sym_name)

    def __lt__(self, other):
        return self.m < other.m


class Rate:
    """ a single Reaclib rate, which can be composed of multiple sets """
    def __init__(self, rfile=None, rfile_path=None, chapter=None, original_source=None,
                 reactants=None, products=None, sets=None, labelprops=None, Q=None):
        """ rfile can be either a string specifying the path to a rate file or
        an io.StringIO object from which to read rate information. """

        self.rfile_path = rfile_path
        self.rfile = None

        if type(rfile) == str:
            self.rfile_path = _find_rate_file(rfile)
            self.rfile = os.path.basename(rfile)

        self.chapter = chapter    # the Reaclib chapter for this reaction
        self.original_source = original_source   # the contents of the original rate file
        self.fname = None

        if reactants:
            self.reactants = reactants
        else:
            self.reactants = []

        if products:
            self.products = products
        else:
            self.products = []

        if sets:
            self.sets = sets
        else:
            self.sets = []

        self.labelprops = labelprops

        self.label = None
        self.resonant = None
        self.resonance_combined = None
        self.weak = None
        self.weak_type = None
        self.reverse = None
        self.tabular = None

        self.Q = Q

        if type(rfile) == str:
            # read in the file, parse the different sets and store them as
            # SingleSet objects in sets[]
            f = open(self.rfile_path)
        elif type(rfile) == io.StringIO:
            # Set f to the io.StringIO object
            f = rfile
        else:
            f = None

        if f:
            self._read_from_file(f)
            f.close()
        else:
            self._set_label_properties()

        self._set_rhs_properties()
        self._set_screening()
        self._set_print_representation()

        if self.tabular:
            self.get_tabular_rate()

    def __repr__(self):
        return self.string

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        """ Determine whether two Rate objects are equal.
        They are equal if they contain identical reactants and products and
        if they contain the same SingleSet sets and if their chapters are equal."""
        x = True

        x = x and (self.chapter == other.chapter)
        x = x and (self.reactants == other.reactants)
        x = x and (self.products == other.products)
        x = x and (len(self.sets) == len(other.sets))

        for si in self.sets:
            scomp = False
            for sj in other.sets:
                if si == sj:
                    scomp = True
                    break
            x = x and scomp

        return x

    def __lt__(self, other):
        """sort such that lightest reactants come first, and then look at products"""

        # this sort will make two nuclei with the same A be in order of Z
        # (assuming there are no nuclei with A > 999
        # we want to compare based on the heaviest first, so we reverse

        self_react_sorted = sorted(self.reactants, key=lambda x: 1000*x.A + x.Z, reverse=True)
        other_react_sorted = sorted(other.reactants, key=lambda x: 1000*x.A + x.Z, reverse=True)

        if self_react_sorted != other_react_sorted:
            # reactants are different, so now we can check them
            for srn, orn in zip(self_react_sorted, other_react_sorted):
                if not srn == orn:
                    return srn < orn
        else:
            # reactants are the same, so consider products
            self_prod_sorted = sorted(self.products, key=lambda x: 1000*x.A + x.Z, reverse=True)
            other_prod_sorted = sorted(other.products, key=lambda x: 1000*x.A + x.Z, reverse=True)

            for spn, opn in zip(self_prod_sorted, other_prod_sorted):
                if not spn == opn:
                    return spn < opn

        # if we made it here, then the rates are the same
        return True

    def __add__(self, other):
        """Combine the sets of two Rate objects if they describe the same
           reaction. Must be Reaclib rates."""
        assert self.reactants == other.reactants
        assert self.products == other.products
        assert self.chapter == other.chapter
        assert isinstance(self.chapter, int)
        assert self.label == other.label
        assert self.weak == other.weak
        assert self.weak_type == other.weak_type
        assert self.tabular == other.tabular
        assert self.reverse == other.reverse

        if self.resonant != other.resonant:
            self._labelprops_combine_resonance()
        new_rate = Rate(chapter=self.chapter,
                        original_source='\n'.join([self.original_source,
                                                   other.original_source]),
                        reactants=self.reactants,
                        products=self.products,
                        sets=self.sets + other.sets,
                        labelprops=self.labelprops,
                        Q=self.Q)
        return new_rate

    def _set_label_properties(self, labelprops=None):
        """ Calls _update_resonance_combined and then
            _update_label_properties. """
        if labelprops:
            self.labelprops = labelprops

        # Update labelprops based on the Sets in this Rate
        # to set the resonance_combined flag properly
        self._update_resonance_combined()
        self._update_label_properties()

    def _update_resonance_combined(self):
        """ Checks the Sets in this Rate and updates the
            resonance_combined flag as well as
            self.labelprops[4] """
        sres = [s.resonant for s in self.sets]
        if True in sres and False in sres:
            self._labelprops_combine_resonance()
        else:
            self.resonance_combined = False

    def _labelprops_combine_resonance(self):
        """ Update self.labelprops[4] = 'c'.
            Also set the resonance_combined flag. """
        llp = list(self.labelprops)
        llp[4] = 'c'
        self.labelprops = ''.join(llp)
        self.resonance_combined = True

    def _update_label_properties(self):
        """ Set label and flags indicating Rate is resonant,
            weak, or reverse. """
        assert isinstance(self.labelprops, str)
        try:
            assert len(self.labelprops) == 6
        except:
            assert self.labelprops == 'tabular'
            self.label = 'tabular'
            self.resonant = False
            self.resonance_combined = False
            self.weak = False # The tabular rate might or might not be weak
            self.weak_type = None
            self.reverse = False
            self.tabular = True
        else:
            self.label = self.labelprops[0:4]
            self.resonant = self.labelprops[4] == 'r'
            self.weak = self.labelprops[4] == 'w'
            if self.weak:
                if self.label.strip() == 'ec' or self.label.strip() == 'bec':
                    self.weak_type = 'electron_capture'
                else:
                    self.weak_type = self.label.strip().replace('+','_pos_').replace('-','_neg_')
            else:
                self.weak_type = None
            self.reverse = self.labelprops[5] == 'v'
            self.tabular = False

    def _read_from_file(self, f):
        """ given a file object, read rate data from the file. """
        lines = f.readlines()
        f.close()

        self.original_source = "".join(lines)

        # first line is the chapter
        self.chapter = lines[0].strip()
        # catch table prescription
        if self.chapter != "t":
            self.chapter = int(self.chapter)

        # remove any blank lines
        set_lines = [l for l in lines[1:] if not l.strip() == ""]

        if self.chapter == "t":
            # e1 -> e2, Tabulated
            s1 = set_lines.pop(0)
            s2 = set_lines.pop(0)
            s3 = set_lines.pop(0)
            s4 = set_lines.pop(0)
            s5 = set_lines.pop(0)
            f = s1.split()
            try:
                self.reactants.append(Nucleus(f[0]))
                self.products.append(Nucleus(f[1]))
            except:
                print(f'Nucleus objects not be identified in {self.original_source}')
                raise

            self.table_file = s2.strip()
            self.table_header_lines = int(s3.strip())
            self.table_rhoy_lines = int(s4.strip())
            self.table_temp_lines = int(s5.strip())
            self.table_num_vars = 6 # Hard-coded number of variables in tables for now.
            self.table_index_name = f'j_{self.reactants[0]}_{self.products[0]}'
            self.labelprops = 'tabular'
            self._set_label_properties()

        else:
            # the rest is the sets
            first = 1
            while len(set_lines) > 0:
                # check for a new chapter id in case of Reaclib v2 format
                check_chapter = set_lines[0].strip()
                try:
                    # see if there is a chapter number preceding the set
                    check_chapter = int(check_chapter)
                except:
                    # there was no chapter number, proceed reading a set
                    pass
                else:
                    # there was a chapter number so check that the chapter number
                    # is the same as the first set in this rate file
                    try:
                        assert check_chapter == self.chapter
                    except:
                        print(f'ERROR: read chapter {check_chapter}, expected chapter {self.chapter} for this rate set.')
                        raise
                    else:
                        # get rid of chapter number so we can read a rate set
                        set_lines.pop(0)

                # sets are 3 lines long
                s1 = set_lines.pop(0)
                s2 = set_lines.pop(0)
                s3 = set_lines.pop(0)

                # first line of a set has up to 6 nuclei, then the label,
                # and finally the Q value

                # get rid of first 5 spaces
                s1 = s1[5:]

                # next follows 6 fields of 5 characters containing nuclei
                # the 6 fields are padded with spaces
                f = []
                for i in range(6):
                    ni = s1[:5]
                    s1 = s1[5:]
                    ni = ni.strip()
                    if ni:
                        f.append(ni)

                # next come 8 spaces, so get rid of them
                s1 = s1[8:]

                # next is a 4-character set label and 2 character flags
                labelprops = s1[:6]
                s1 = s1[6:]

                # next come 3 spaces
                s1 = s1[3:]

                # next comes a 12 character Q value followed by 10 spaces
                Q = float(s1.strip())

                if first:
                    self.Q = Q

                    try:
                        # what's left are the nuclei -- their interpretation
                        # depends on the chapter
                        if self.chapter == 1:
                            # e1 -> e2
                            self.reactants.append(Nucleus(f[0]))
                            self.products.append(Nucleus(f[1]))

                        elif self.chapter == 2:
                            # e1 -> e2 + e3
                            self.reactants.append(Nucleus(f[0]))
                            self.products += [Nucleus(f[1]), Nucleus(f[2])]

                        elif self.chapter == 3:
                            # e1 -> e2 + e3 + e4
                            self.reactants.append(Nucleus(f[0]))
                            self.products += [Nucleus(f[1]), Nucleus(f[2]), Nucleus(f[3])]

                        elif self.chapter == 4:
                            # e1 + e2 -> e3
                            self.reactants += [Nucleus(f[0]), Nucleus(f[1])]
                            self.products.append(Nucleus(f[2]))

                        elif self.chapter == 5:
                            # e1 + e2 -> e3 + e4
                            self.reactants += [Nucleus(f[0]), Nucleus(f[1])]
                            self.products += [Nucleus(f[2]), Nucleus(f[3])]

                        elif self.chapter == 6:
                            # e1 + e2 -> e3 + e4 + e5
                            self.reactants += [Nucleus(f[0]), Nucleus(f[1])]
                            self.products += [Nucleus(f[2]), Nucleus(f[3]), Nucleus(f[4])]

                        elif self.chapter == 7:
                            # e1 + e2 -> e3 + e4 + e5 + e6
                            self.reactants += [Nucleus(f[0]), Nucleus(f[1])]
                            self.products += [Nucleus(f[2]), Nucleus(f[3]),
                                              Nucleus(f[4]), Nucleus(f[5])]

                        elif self.chapter == 8:
                            # e1 + e2 + e3 -> e4
                            self.reactants += [Nucleus(f[0]), Nucleus(f[1]), Nucleus(f[2])]
                            self.products.append(Nucleus(f[3]))

                        elif self.chapter == 9:
                            # e1 + e2 + e3 -> e4 + e5
                            self.reactants += [Nucleus(f[0]), Nucleus(f[1]), Nucleus(f[2])]
                            self.products += [Nucleus(f[3]), Nucleus(f[4])]

                        elif self.chapter == 10:
                            # e1 + e2 + e3 + e4 -> e5 + e6
                            self.reactants += [Nucleus(f[0]), Nucleus(f[1]),
                                               Nucleus(f[2]), Nucleus(f[3])]
                            self.products += [Nucleus(f[4]), Nucleus(f[5])]

                        elif self.chapter == 11:
                            # e1 -> e2 + e3 + e4 + e5
                            self.reactants.append(Nucleus(f[0]))
                            self.products += [Nucleus(f[1]), Nucleus(f[2]),
                                              Nucleus(f[3]), Nucleus(f[4])]
                        else:
                            print(f'Chapter could not be identified in {self.original_source}')
                            assert isinstance(self.chapter, int) and self.chapter <= 11
                    except:
                        # print('Error parsing Rate from {}'.format(self.original_source))
                        raise

                    first = 0

                # the second line contains the first 4 coefficients
                # the third lines contains the final 3
                # we can't just use split() here, since the fields run into one another
                n = 13  # length of the field
                a = [s2[i:i+n] for i in range(0, len(s2), n)]
                a += [s3[i:i+n] for i in range(0, len(s3), n)]

                a = [float(e) for e in a if not e.strip() == ""]
                self.sets.append(SingleSet(a, labelprops=labelprops))
                self._set_label_properties(labelprops)

    def _set_rhs_properties(self):
        """ compute statistical prefactor and density exponent from the reactants. """
        self.prefactor = 1.0  # this is 1/2 for rates like a + a (double counting)
        self.inv_prefactor = 1
        for r in set(self.reactants):
            self.inv_prefactor = self.inv_prefactor * np.math.factorial(self.reactants.count(r))
        self.prefactor = self.prefactor/float(self.inv_prefactor)
        self.dens_exp = len(self.reactants)-1
        if (self.weak_type == 'electron_capture' and not self.tabular):
            self.dens_exp = self.dens_exp + 1

    def _set_screening(self):
        """ determine if this rate is eligible for screening and the nuclei to use. """
        # Tells if this rate is eligible for screening
        # using screenz.f90 provided by StarKiller Microphysics.
        # If not eligible for screening, set to None
        # If eligible for screening, then
        # Rate.ion_screen is a 2-element (3 for 3-alpha) list of Nucleus objects for screening
        self.ion_screen = []
        nucz = [q for q in self.reactants if q.Z != 0]
        if len(nucz) > 1:
            nucz.sort(key=lambda x: x.Z)
            self.ion_screen = []
            self.ion_screen.append(nucz[0])
            self.ion_screen.append(nucz[1])
            if len(nucz) == 3:
                self.ion_screen.append(nucz[2])

        # if the rate is a reverse rate, via detailed balance, then we
        # might actually want to compute the screening based on the
        # reactants of the forward rate that was used in the detailed
        # balance.  Rate.symmetric_screen is what should be used in
        # the screening in this case
        self.symmetric_screen = []
        if self.reverse:
            nucz = [q for q in self.products if q.Z != 0]
            if len(nucz) > 1:
                nucz.sort(key=lambda x: x.Z)
                self.symmetric_screen = []
                self.symmetric_screen.append(nucz[0])
                self.symmetric_screen.append(nucz[1])
                if len(nucz) == 3:
                    self.symmetric_screen.append(nucz[2])
        else:
            self.symmetric_screen = self.ion_screen

    def _set_print_representation(self):
        """ compose the string representations of this Rate. """
        self.string = ""
        self.pretty_string = r"$"

        # put p, n, and alpha second
        treactants = []
        for n in self.reactants:
            if n.raw not in ["p", "he4", "n"]:
                treactants.insert(0, n)
            else:
                treactants.append(n)

        for n, r in enumerate(treactants):
            self.string += f"{r}"
            self.pretty_string += fr"{r.pretty}"
            if not n == len(self.reactants)-1:
                self.string += " + "
                self.pretty_string += r" + "

        self.string += " --> "
        self.pretty_string += r" \rightarrow "

        for n, p in enumerate(self.products):
            self.string += f"{p}"
            self.pretty_string += fr"{p.pretty}"
            if not n == len(self.products)-1:
                self.string += " + "
                self.pretty_string += r" + "

        self.pretty_string += r"$"

        if not self.fname:
            # This is used to determine which rates to detect as the same reaction
            # from multiple sources in a Library file, so it should not be unique
            # to a given source, e.g. wc12, but only unique to the reaction.
            reactants_str = '_'.join([repr(nuc) for nuc in self.reactants])
            products_str = '_'.join([repr(nuc) for nuc in self.products])
            self.fname = f'{reactants_str}__{products_str}'
            if self.weak:
                self.fname = self.fname + f'__weak__{self.weak_type}'

    def get_rate_id(self):
        """ Get an identifying string for this rate.
        Don't include resonance state since we combine resonant and
        non-resonant versions of reactions. """

        srev = ''
        if self.reverse:
            srev = 'reverse'

        sweak = ''
        if self.weak:
            sweak = 'weak'

        ssrc = 'reaclib'
        if self.tabular:
            ssrc = 'tabular'

        return f'{self.__repr__()} <{self.label.strip()}_{ssrc}_{sweak}_{srev}>'

    def heaviest(self):
        """
        Return the heaviest nuclide in this Rate.

        If two nuclei are tied in mass number, return the one with the
        lowest atomic number.
        """
        nuc = self.reactants[0]
        for n in self.reactants + self.products:
            if n.A > nuc.A or (n.A == nuc.A and n.Z < nuc.Z):
                nuc = n
        return nuc

    def lightest(self):
        """
        Return the lightest nuclide in this Rate.

        If two nuclei are tied in mass number, return the one with the
        highest atomic number.
        """
        nuc = self.reactants[0]
        for n in self.reactants + self.products:
            if n.A < nuc.A or (n.A == nuc.A and n.Z > nuc.Z):
                nuc = n
        return nuc

    def get_tabular_rate(self):
        """read the rate data from .dat file """

        # find .dat file and read it
        self.table_path = _find_rate_file(self.table_file)
        tabular_file = open(self.table_path)
        t_data = tabular_file.readlines()
        tabular_file.close()

        # delete header lines
        del t_data[0:self.table_header_lines]

        # change the list ["1.23 3.45 5.67\n"] into the list ["1.23","3.45","5.67"]
        t_data2d = []
        for tt in t_data:
            t_data2d.append(re.split(r"[ ]", tt.strip('\n')))

        # delete all the "" in each element of data1
        for tt2d in t_data2d:
            while '' in tt2d:
                tt2d.remove('')

        while [] in t_data2d:
            t_data2d.remove([])

        self.tabular_data_table = np.array(t_data2d)

    def eval(self, T, rhoY = None):
        """ evauate the reaction rate for temperature T """

        if self.tabular:
            data = self.tabular_data_table.astype(np.float)
            # find the nearest value of T and rhoY in the data table
            T_nearest = (data[:,1])[np.abs((data[:,1]) - T).argmin()]
            rhoY_nearest = (data[:,0])[np.abs((data[:,0]) - rhoY).argmin()]
            inde = np.where((data[:,1]==T_nearest)&(data[:,0]==rhoY_nearest))[0][0]
            r = data[inde][5]

        else:
            tf = Tfactors(T)
            r = 0.0
            for s in self.sets:
                f = s.f()
                r += f(tf)

        return r

    def get_rate_exponent(self, T0):
        """
        for a rate written as a power law, r = r_0 (T/T0)**nu, return
        nu corresponding to T0
        """

        # nu = dln r /dln T, so we need dr/dT
        r1 = self.eval(T0)
        dT = 1.e-8*T0
        r2 = self.eval(T0 + dT)

        drdT = (r2 - r1)/dT
        return (T0/r1)*drdT

    def plot(self, Tmin=1.e8, Tmax=1.6e9, rhoYmin=3.9e8, rhoYmax=2.e9):
        """plot the rate's temperature sensitivity vs temperature"""

        if self.tabular:
            data = self.tabular_data_table.astype(np.float) # convert from str to float

            inde1 = data[:,1]<=Tmax
            inde2 = data[:,1]>=Tmin
            inde3 = data[:,0]<=rhoYmax
            inde4 = data[:,0]>=rhoYmin
            data_heatmap = data[inde1&inde2&inde3&inde4].copy()

            rows, row_pos = np.unique(data_heatmap[:, 0], return_inverse=True)
            cols, col_pos = np.unique(data_heatmap[:, 1], return_inverse=True)
            pivot_table = np.zeros((len(rows), len(cols)), dtype=data_heatmap.dtype)
            try:
                pivot_table[row_pos, col_pos] = np.log10(data_heatmap[:, 5])
            except ValueError:
                print("Divide by zero encountered in log10\nChange the scale of T or rhoY")

            _, ax = plt.subplots(figsize=(10,10))
            im = ax.imshow(pivot_table, cmap='jet')
            plt.colorbar(im)

            plt.xlabel("$T$ [K]")
            plt.ylabel("$\\rho Y$ [g/cm$^3$]")
            ax.set_title(fr"{self.pretty_string}"+
                         "\n"+"electron-capture/beta-decay rate in log10(1/s)")
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(rows)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols)
            plt.setp(ax.get_xticklabels(), rotation=90, ha="right",rotation_mode="anchor")
            plt.gca().invert_yaxis()
            plt.show()

        else:
            temps = np.logspace(np.log10(Tmin), np.log10(Tmax), 100)
            r = np.zeros_like(temps)

            for n, T in enumerate(temps):
                r[n] = self.eval(T)

            plt.loglog(temps, r)
            plt.xlabel(r"$T$")

            if self.dens_exp == 0:
                plt.ylabel(r"\tau")
            elif self.dens_exp == 1:
                plt.ylabel(r"$N_A <\sigma v>$")
            elif self.dens_exp == 2:
                plt.ylabel(r"$N_A^2 <n_a n_b n_c v>$")

            plt.title(fr"{self.pretty_string}")
            plt.show()

class ChemComposition:
    """a composition holds the mass fractions of the chemspecies in a network
    -- useful for evaluating the rates

    FOR NOW, I ONLY CHANGED SELF.SYMPY TO TREAT COMPOSITIONS AS MASS FRACTIONS
    REST OF THE CODE TREATS THEM AS NUMBER DENSITIES

    """
    def __init__(self, specie, small=1.e-40):
        """specie is an iterable of the specie (ChemSpecie objects) in the network"""
        if not isinstance(specie[0], ChemSpecie):
            raise ValueError("must supply an iterable of ChemSpecie object")
        self.X = {k: small for k in specie}
        self.Y = {k.sym_name: k.sym_name for k in specie}

    def set_all(self, xval):
        """ set all species to a particular value """
        for k in self.X:
            self.X[k] = xval
        return self.X

    def set_specie_massfrac(self, xval):
        """ set specie name to the mass fraction xval """

        if len(self.X) != len(xval):
            raise ValueError("length of species array does not match length of mass fractions array")

        #need a separate counter for xval tuple
        i = 0
        for k in self.X:
            self.X[k] = xval[i]
            i = i+1
        
        self.normalize()
        return self.X

    def set_specie_numberdens(self, nval):
        """ set specie name to the number density nval """

        if len(self.X) != len(nval):
            raise ValueError("length of species array does not match length of number densities array")

        #need a separate counter for xval tuple
        i = 0
        for k in self.X:
            self.X[k] = nval[i]
            i = i+1
        
        return self.X

    def normalize(self):
        """ normalize the mass fractions to sum to 1 """
        X_sum = sum(self.X[k] for k in self.X)

        for k in self.X:
            self.X[k] /= X_sum
    
        return self.X

    def get_specie_numberdens(self, xval, rho):

        massfracs = self.set_specie_massfrac(xval)
        for k in self.X:
            self.X[k] = massfracs[k] * rho / k.m

        return self.X

    def sympy(self):
        #for k in self.Y:
        #    self.Y[k] = k.sym_name

        return self.Y

    def __str__(self):
        ostr = ""
        for k in self.X:
            ostr += f"  X({k}) : {self.X[k]}\n"
        return ostr


class ChemRate:
    def __init__(self, reactants=[], products=[]):
        self.reactants = reactants

        self.products = products
        #self._set_print_representation()

        if {ChemSpecie('h'), ChemSpecie('elec')} == set(self.reactants) and Counter(self.products)[ChemSpecie('hp')] == 1 and Counter(self.products)[ChemSpecie('elec')] == 2:
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                #reaction 1
                rate = self.get_small(composition) + np.exp(-32.71396786+13.5365560*lnTe-5.73932875*(lnTe**2)+1.56315498*(lnTe**3)-0.28770560*(lnTe**4)+3.48255977e-2*(lnTe**5)-2.63197617e-3*(lnTe**6)+1.11954395e-4*(lnTe**7)-2.03914985e-6*(lnTe**8))            
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hp'), ChemSpecie('elec')} == set(self.reactants) and {ChemSpecie('h')} == set(self.products):
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                #reactions 2 and 3
                if Te <= 5.5e3:
                    rate = self.get_small(composition) + 3.92e-13*invTe**0.6353
                else:
                    rate = self.get_small(composition) + np.exp(-28.61303380689232-0.7241125657826851*lnTe-0.02026044731984691*lnTe**2-0.002380861877349834*lnTe**3-0.0003212605213188796*lnTe**4-0.00001421502914054107*lnTe**5+4.989108920299513e-6*lnTe**6+5.755614137575758e-7*lnTe**7-1.856767039775261e-8*lnTe**8-3.071135243196595e-9*lnTe**9)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('he'), ChemSpecie('elec')} == set(self.reactants) and Counter(self.products)[ChemSpecie('hep')] == 1 and Counter(self.products)[ChemSpecie('elec')] == 2:
            #reaction 4
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + np.exp(-44.09864886+23.91596563*lnTe-10.7532302*(lnTe**2)+3.05803875*(lnTe**3)-0.56851189*(lnTe**4)+6.79539123e-2*(lnTe**5)-5.00905610e-3*(lnTe**6)+2.06723616e-4*(lnTe**7)-3.64916141e-6*(lnTe**8))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep'), ChemSpecie('elec')} == set(self.reactants) and {ChemSpecie('he')} == set(self.products):
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                #reactions 5 and 6
                if Te <= 9.28e3:
                    rate = self.get_small(composition) + 3.92e-13*invTe**0.6353
                else:
                    rate = self.get_small(composition) + 1.54e-9*(1.0+0.30/np.exp(8.099328789667*invTe))/(np.exp(40.49664394833662*invTe)*Te**1.50)+3.92e-13/Te**0.6353
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep'), ChemSpecie('elec')} == set(self.reactants) and Counter(self.products)[ChemSpecie('hepp')] == 1 and Counter(self.products)[ChemSpecie('elec')] == 2:
            #reaction 7
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + np.exp(-68.71040990212001+43.93347632635*lnTe-18.48066993568*lnTe**2+4.701626486759002*lnTe**3-0.7692466334492*lnTe**4+0.08113042097303*lnTe**5-0.005324020628287001*lnTe**6+0.0001975705312221*lnTe**7-3.165581065665e-6*lnTe**8)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hepp'), ChemSpecie('elec')} == set(self.reactants) and {ChemSpecie('hep')} == set(self.products):
            #reaction 8
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1.891e-10/(np.sqrt(Tgas/9.37)*(1.+np.sqrt(Tgas/9.37))**0.2476*(1.+np.sqrt(Tgas/2.774e6))**1.7524)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h'), ChemSpecie('elec')} == set(self.reactants) and {ChemSpecie('hm')} == set(self.products):
            #reaction 9
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1.4e-18*Tgas**0.928*np.exp(-Tgas/16200.)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm'), ChemSpecie('h')} == set(self.reactants) and {ChemSpecie('h2'), ChemSpecie('elec')} == set(self.products):
            #reaction 10
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                a1=1.3500e-09
                a2=9.8493e-02
                a3=3.2852e-01
                a4=5.5610e-01
                a5=2.7710e-07
                a6=2.1826e+00
                a7=6.1910e-03
                a8=1.0461e+00
                a9=8.9712e-11
                a10=3.0424e+00
                a11=3.2576e-14
                a12=3.7741e+00
                rate = self.get_small(composition) + a1*(Tgas**a2+a3*Tgas**a4+a5*Tgas**a6)/(1.+a7*Tgas**a8+a9*Tgas**a10+a11*Tgas**a12)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h'), ChemSpecie('hp')} == set(self.reactants) and {ChemSpecie('h2p')} == set(self.products):
            #reactions 11 and 12
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                if Tgas < 30:
                    rate = self.get_small(composition) + 2.10e-20*(Tgas/30.)**(-0.15)
                else:
                    rate = self.get_small(composition) + 10**(-18.20-3.194*np.log10(Tgas)+1.786*(np.log10(Tgas))**2-0.2072*(np.log10(Tgas))**3)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p'), ChemSpecie('h')} == set(self.reactants) and {ChemSpecie('hp'), ChemSpecie('h2')} == set(self.products):
            #reaction 13
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 6.0e-10
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2'), ChemSpecie('hp')} == set(self.reactants) and {ChemSpecie('h2p'), ChemSpecie('h')} == set(self.products):
            #reaction 14
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                if Tgas >= 1e2 and Tgas <= 3e4:
                    asav = 2.1237150e4
                    bsav1=-3.3232183e-7
                    bsav2= 3.3735382e-7
                    bsav3=-1.4491368e-7 
                    bsav4= 3.4172805e-8 
                    bsav5=-4.7813728e-9 
                    bsav6= 3.9731542e-10
                    bsav7=-1.8171411e-11
                    bsav8= 3.5311932e-13
                    sumsav=bsav1+bsav2*np.log(Tgas)+bsav3*(np.log(Tgas))**2+bsav4*(np.log(Tgas))**3+bsav5*(np.log(Tgas))**4+bsav6*(np.log(Tgas))**5+bsav7*(np.log(Tgas))**6+bsav8*(np.log(Tgas))**7
                    rate = self.get_small(composition) + sumsav*np.exp(-asav*invT)
                else:
                    rate = 0.0
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2'), ChemSpecie('elec')} == set(self.reactants) and {ChemSpecie('h'), ChemSpecie('hm')} == set(self.products):
            #reaction 15
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 3.55e1*Tgas**(-2.28)*np.exp(-46707./Tgas)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2'), ChemSpecie('elec')} == set(self.reactants) and Counter(self.products)[ChemSpecie('h')] == 2 and Counter(self.products)[ChemSpecie('elec')] == 1:
            #reaction 16
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 4.38e-10*np.exp(-102000./Tgas)*Tgas**(0.35)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2'), ChemSpecie('h')} == set(self.reactants) and Counter(self.products)[ChemSpecie('h')] == 3:
            #reaction 17
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):


                k_CIDm = np.zeros((2, 21))
                k_CIDm[0] = (-178.4239, -68.42243, 43.20243, -4.633167, \
                             69.70086, 40870.38, -23705.70, 128.8953, -53.91334, \
                             5.315517, -19.73427, 16780.95, -25786.11, 14.82123, \
                             -4.890915, 0.4749030, -133.8283, -1.164408, 0.8227443, \
                             0.5864073, -2.056313)

                k_CIDm[1] = (-142.7664, 42.70741, -2.027365, -0.2582097, \
                              21.36094, 27535.31, -21467.79, 60.34928, -27.43096, \
                              2.676150, -11.28215, 14254.55, -23125.20, 9.305564, \
                             -2.464009, 0.1985955, 743.0600, -1.174242, 0.7502286, \
                             0.2358848, 2.937507)

                n_H  = self.get_Hnuclei(composition)
                logT = np.log10(Tgas)
                invT = 1.0/Tgas
                logT2 = logT*logT
                logT3 = logT2*logT
                logTv = np.array([1.0, logT, logT2, logT3])
                k_CID = 0.

                i = 0
                while i < 2:
                    logk_h1 = k_CIDm[i,0]*logTv[0] + k_CIDm[i,1]*logTv[1] + \
                              k_CIDm[i,2]*logTv[2] + k_CIDm[i,3]*logTv[3] + \
                              k_CIDm[i,4]*np.log10(1.0+k_CIDm[i,5]*invT)

                    logk_h2 = k_CIDm[i,6]*invT

                    logk_l1 = k_CIDm[i,7]*logTv[0] + k_CIDm[i,8]*logTv[1] + \
                              k_CIDm[i,9]*logTv[2] + k_CIDm[i,10]*np.log10(1.0+k_CIDm[i,11]*invT)

                    logk_l2 = k_CIDm[i,12]*invT

                    logn_c1 = k_CIDm[i,13]*logTv[0] + k_CIDm[i,14]*logTv[1] + \
                              k_CIDm[i,15]*logTv[2] + k_CIDm[i,16]*invT

                    logn_c2 = k_CIDm[i,17] + logn_c1

                    p = k_CIDm[i,18] + k_CIDm[i,19]*np.exp(-Tgas/1.850e3) + \
                        k_CIDm[i,20]*np.exp(-Tgas/4.40e2)

                    n_c1 = 1e1**(logn_c1)
                    n_c2 = 1e1**(logn_c2)

                    logk_CID = logk_h1 - (logk_h1 - logk_l1) / (1.0 + (n_H/n_c1)**p) + \
                               logk_h2 - (logk_h2 - logk_l2) / (1.0 + (n_H/n_c2)**p)

                    k_CID = k_CID + 1.e1**logk_CID

                    i += 1

                return self.get_small(composition) + k_CID

                #rate = 0. #self.dissH2_Martin96(Tgas, composition)
                #return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm'), ChemSpecie('elec')} == set(self.reactants) and Counter(self.products)[ChemSpecie('h')] == 1 and Counter(self.products)[ChemSpecie('elec')] == 2:
            #reaction 18
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + np.exp(-18.01849334273+2.360852208681*lnTe-0.2827443061704*lnTe**2+0.01623316639567*lnTe**3-0.03365012031362999*lnTe**4+0.01178329782711*lnTe**5-0.001656194699504*lnTe**6+0.0001068275202678*lnTe**7-2.631285809207e-6*lnTe**8)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm'), ChemSpecie('h')} == set(self.reactants) and Counter(self.products)[ChemSpecie('h')] == 2 and Counter(self.products)[ChemSpecie('elec')] == 1:
            #reactions 19 and 20
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                if Tgas <= 1.16e3:
                    rate = self.get_small(composition) + 2.56e-9*Te**1.78186
                else:
                    rate = self.get_small(composition) + np.exp(-20.37260896533324+1.139449335841631*lnTe-0.1421013521554148*lnTe**2+0.00846445538663*lnTe**3-0.0014327641212992*lnTe**4+0.0002012250284791*lnTe**5+0.0000866396324309*lnTe**6-0.00002585009680264*lnTe**7+2.4555011970392e-6*lnTe**8-8.06838246118e-8*lnTe**9)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm'), ChemSpecie('hp')} == set(self.reactants) and Counter(self.products)[ChemSpecie('h')] == 2:
            #reaction 21
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                if Tgas >= 1e1 and Tgas <= 1e5:
                    rate = self.get_small(composition) + (2.96e-6/np.sqrt(Tgas)-1.73e-9+2.50e-10*np.sqrt(Tgas)-7.77e-13*Tgas)
                else:
                    rate = 0.0
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm'), ChemSpecie('hp')} == set(self.reactants) and {ChemSpecie('h2p'), ChemSpecie('elec')} == set(self.products):
            #reaction 22
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1e-8*Tgas**(-0.4)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p'), ChemSpecie('elec')} == set(self.reactants) and Counter(self.products)[ChemSpecie('h')] == 2:
            #reaction 23
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                if Tgas <= 1e4:
                    rate = self.get_small(composition) + 1e6*(4.2278e-14-2.3088e-17*Tgas+7.3428e-21*Tgas**2-7.5474e-25*Tgas**3+3.3468e-29*Tgas**4-5.528e-34*Tgas**5)
                else:
                    rate = 0.0
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p'), ChemSpecie('hm')} == set(self.reactants) and {ChemSpecie('h'), ChemSpecie('h2')} == set(self.products):
            #reaction 24
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 5e-7*np.sqrt(1.e2*invT)
                return rate
            self.rate_function = rate_function

        elif Counter(self.reactants)[ChemSpecie('h')] == 3 and {ChemSpecie('h2'), ChemSpecie('h')} == set(self.products):
            #reaction 25
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 6e-32*Tgas**(-0.25)+2e-31*Tgas**(-0.5)
                return rate
            self.rate_function = rate_function

        elif Counter(self.reactants)[ChemSpecie('h')] == 2 and Counter(self.reactants)[ChemSpecie('h2')] == 1 and Counter(self.products)[ChemSpecie('h2')] == 2:
            #reaction 26
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + (6e-32*Tgas**(-0.25)+2e-31*Tgas**(-0.5))/8.0
                return rate
            self.rate_function = rate_function

        elif Counter(self.reactants)[ChemSpecie('h2')] == 2 and Counter(self.products)[ChemSpecie('h2')] == 1 and Counter(self.products)[ChemSpecie('h')] == 2:
            #reaction 27
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                Hnuclei = self.get_Hnuclei(composition)
                kl21 = 1.18e-10*np.exp(-6.95e4*invT)
                kh21 = 8.125e-8*Tgas**(-0.5)*np.exp(-5.2e4*invT)*(1.0-np.exp(-6e3*invT))
                ncr21 = 1e1**(4.845-1.3*np.log10(Tgas*1e-4)+1.62*(np.log10(Tgas*1e-4))**2)
                a21=1.0/(1.0+(Hnuclei/ncr21))
                rate = self.get_small(composition) + kh21**(1.0 - a21)*kl21**a21
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep'), ChemSpecie('h')} == set(self.reactants) and {ChemSpecie('he'), ChemSpecie('hp')} == set(self.products):
            #reaction 28
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1.20e-15*(Tgas/3e2)**0.25
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('he'), ChemSpecie('hp')} == set(self.reactants) and {ChemSpecie('hep'), ChemSpecie('h')} == set(self.products):
            #reactions 29 and 30
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                if Tgas <= 1e4:
                    rate = self.get_small(composition) + 1.26e-9*Tgas**(-0.75)*np.exp(-1.275e5*invT)
                else:
                    rate = self.get_small(composition) + 4e-37*Tgas**(4.74)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2'), ChemSpecie('dp')} == set(self.reactants) and {ChemSpecie('hd'), ChemSpecie('hp')} == set(self.products):
            #reaction 31
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1e-9*(0.417+0.846*np.log10(Tgas)-0.137*(np.log10(Tgas))**2)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hd'), ChemSpecie('hp')} == set(self.reactants) and {ChemSpecie('h2'), ChemSpecie('dp')} == set(self.products):
            #reaction 32
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1e-9*np.exp(-4.57e2*invT)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2'), ChemSpecie('d')} == set(self.reactants) and {ChemSpecie('hd'), ChemSpecie('h')} == set(self.products):
            #reactions 33 and 34
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                if Tgas <= 1.167479642374226e3:
                    rate = self.get_small(composition) + 10**(-56.4737+5.88886*np.log10(Tgas)+7.19692*(np.log10(Tgas))**2+2.25069*(np.log10(Tgas))**3-2.16903*(np.log10(Tgas))**4+0.317887*(np.log10(Tgas))**5)
                else:
                    rate = self.get_small(composition) + 3.17e-10*np.exp(-5207.*invT)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hd'), ChemSpecie('h')} == set(self.reactants) and {ChemSpecie('h2'), ChemSpecie('d')} == set(self.products):
            #reaction 35
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                if Tgas > 2e2:
                    rate = self.get_small(composition) + 5.25e-11*np.exp(-4430.*invT+1.739e5*(invT)**2)
                else:
                    rate = 0.0
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('d'), ChemSpecie('hm')} == set(self.reactants) and {ChemSpecie('hd'), ChemSpecie('elec')} == set(self.products):
            #reaction 36
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1.5e-9*(T32)**(-0.1)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hp'), ChemSpecie('d')} == set(self.reactants) and {ChemSpecie('h'), ChemSpecie('dp')} == set(self.products):
            #reaction 37
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                if Tgas >= 5e1:
                    rate = self.get_small(composition) + 2e-10*Tgas**(0.402)*np.exp(-37.1*invT)-3.31e-17*Tgas**(1.48)
                else:
                    rate = 0.0
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h'), ChemSpecie('dp')} == set(self.reactants) and {ChemSpecie('hp'), ChemSpecie('d')} == set(self.products):
            #reaction 38
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                if Tgas >= 5e1:
                    rate = self.get_small(composition) + 2.06e-10*Tgas**(0.396)*np.exp(-33.0*invT)+2.03e-9*Tgas**(-0.332)
                else:
                    rate = 0.0
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('dp'), ChemSpecie('elec')} == set(self.reactants) and {ChemSpecie('d')} == set(self.products):
            #reaction 39
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 3.6e-12*(Tgas/300)**(-0.75)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h'), ChemSpecie('d')} == set(self.reactants) and {ChemSpecie('hd')} == set(self.products):
            #reaction 40
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1e-25
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hdp'), ChemSpecie('h')} == set(self.reactants) and {ChemSpecie('hd'), ChemSpecie('hp')} == set(self.products):
            #reaction 41
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 6.4e-10
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hp'), ChemSpecie('d')} == set(self.reactants) and {ChemSpecie('hdp')} == set(self.products):
            #reaction 42
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 10.0**(-19.38-1.523*np.log10(Tgas)+1.118*(np.log10(Tgas))**2.0-0.1269*(np.log10(Tgas))**3.0)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h'), ChemSpecie('dp')} == set(self.reactants) and {ChemSpecie('hdp')} == set(self.products):
            #reaction 43
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 10.0**(-19.38-1.523*np.log10(Tgas)+1.118*(np.log10(Tgas))**2.0-0.1269*(np.log10(Tgas))**3.0)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hdp'), ChemSpecie('elec')} == set(self.reactants) and {ChemSpecie('h'), ChemSpecie('d')} == set(self.products):
            #reaction 44
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                invsqrT = 1.0/np.sqrt(Tgas)
                rate = self.get_small(composition) + 7.2e-8*invsqrT
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('d'), ChemSpecie('elec')} == set(self.reactants) and {ChemSpecie('dm')} == set(self.products):
            #reaction 45
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 3e-16*(Tgas/300)**(0.95)*np.exp(-Tgas/9.320e3)
                return rate
            self.rate_function = rate_function
            
        elif {ChemSpecie('dp'), ChemSpecie('dm')} == set(self.reactants) and Counter(self.products)[ChemSpecie('d')] == 2:
            #reaction 46
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 5.7e-8*(Tgas/300)**(-0.5)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('dm'), ChemSpecie('hp')} == set(self.reactants) and {ChemSpecie('h'), ChemSpecie('d')} == set(self.products):
            #reaction 47
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 4.6e-8*(Tgas/300)**(-0.5)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm'), ChemSpecie('d')} == set(self.reactants) and {ChemSpecie('dm'), ChemSpecie('h')} == set(self.products):
            #reaction 48
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 6.4e-9*(Tgas/300)**(0.41)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('dm'), ChemSpecie('h')} == set(self.reactants) and {ChemSpecie('hm'), ChemSpecie('d')} == set(self.products):
            #reaction 49
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 6.4e-9*(Tgas/300)**(0.41)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('dm'), ChemSpecie('h')} == set(self.reactants) and {ChemSpecie('hd'), ChemSpecie('elec')} == set(self.products):
            #reaction 50
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1.5e-9*(Tgas/300)**(-0.1)
                return rate
            self.rate_function = rate_function

        #new reactions, added by Piyush Sharda from SLD98
        elif {ChemSpecie('dp'), ChemSpecie('hm')} == set(self.reactants) and {ChemSpecie('d'), ChemSpecie('h')} == set(self.products):
            #reaction 51
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 4.6e-8*(Tgas/300)**(-0.5)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep'), ChemSpecie('hm')} == set(self.reactants) and {ChemSpecie('he'), ChemSpecie('h')} == set(self.products):
            #reaction 52
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 2.32e-7*((Tgas/300)**(-0.52))*np.exp(Tgas/22400)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep'), ChemSpecie('d')} == set(self.reactants) and {ChemSpecie('he'), ChemSpecie('d+')} == set(self.products):
            #reaction 53
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1.1e-15*(Tgas/300)**(0.25)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep'), ChemSpecie('dm')} == set(self.reactants) and {ChemSpecie('he'), ChemSpecie('d')} == set(self.products):
            #reaction 54
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 3.03e-7*((Tgas/300)**(-0.52))*np.exp(Tgas/22400)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2'), ChemSpecie('hep')} == set(self.reactants) and {ChemSpecie('h2p'), ChemSpecie('he')} == set(self.products):
            #reaction 55
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 7.2e-15
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p'), ChemSpecie('d')} == set(self.reactants) and {ChemSpecie('hdp'), ChemSpecie('h')} == set(self.products):
            #reaction 56
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1.07e-9*((Tgas/300)**(6.2e-2))*np.exp(Tgas/41400)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p'), ChemSpecie('d')} == set(self.reactants) and {ChemSpecie('h2'), ChemSpecie('dp')} == set(self.products):
            #reaction 57
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 6.4e-10
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hdp'), ChemSpecie('h')} == set(self.reactants) and {ChemSpecie('h2p'), ChemSpecie('d')} == set(self.products):
            #reaction 58
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1.0e-9*np.exp(154/Tgas)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2'), ChemSpecie('hep')} == set(self.reactants) and Counter(self.products)[ChemSpecie('he')] == 1 and Counter(self.products)[ChemSpecie('h')] == 2:
            #reaction 59
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 3.7e-14*np.exp(-35/Tgas)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p'), ChemSpecie('hm')} == set(self.reactants) and Counter(self.products)[ChemSpecie('h')] == 3:
            #reaction 60
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 1.4e-7*(Tgas/300)**(-0.5)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep'), ChemSpecie('hd')} == set(self.reactants) and {ChemSpecie('he'), ChemSpecie('hp'), ChemSpecie('d')} == set(self.products):
            #reaction 61
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 5.5e-14*(Tgas/300)**(-0.24)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep'), ChemSpecie('hd')} == set(self.reactants) and {ChemSpecie('he'), ChemSpecie('h'), ChemSpecie('dp')} == set(self.products):
            #reaction 62
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition):
                rate = self.get_small(composition) + 5.5e-14*(Tgas/300)**(-0.24)
                return rate
            self.rate_function = rate_function

        else:
            raise UnsupportedChemRate()


    def get_Hnuclei(self, composition):

        nH = composition[ChemSpecie('hp')] + composition[ChemSpecie('h')] + composition[ChemSpecie('hm')] + \
             composition[ChemSpecie('h2')]*2.0 + composition[ChemSpecie('h2p')]*2.0

        if ChemSpecie('hd') in composition:
            nH += composition[ChemSpecie('hd')] + composition[ChemSpecie('hdp')]

        return nH

    def get_small(self, composition):
        nmax = max(float(max(composition.values())), 1.0)
        small = 1e-40/(nmax**3)
        return small


    def eval(self, T, composition):
        if T <= 0:
            raise UnphysicalTemperature()
        else:
            Tgas = T
            Te = Tgas*8.617343e-5 #CHECK KROME FILES!!!
            invTe = 1.0/Te
            invT = 1.0/T
            lnTe = np.log(Te)
            T32 = Tgas*0.0033333333333333335 #Tgas/(300 K)

            return self.rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition)


    def __repr__(self):
        repstring = str(self.reactants) + ' --> ' + str(self.products)
        return repstring



class SympyChemRate:
    def __init__(self, reactants=[], products=[], massfracs=0):
        import sympy as sp
        self.reactants = reactants

        self.products = products

        self.massfracs = massfracs
        if (self.massfracs != 0) & (self.massfracs != 1):
            raise ValueError('The code only works with mass fractions or number densities!')
        #self._set_print_representation()

        if {ChemSpecie('h').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('hp').sym_name] == 1 and Counter(self.products)[ChemSpecie('elec').sym_name] == 2:
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                #reaction 1
                #rate = self.get_small(composition) + np.exp(-32.71396786+13.5365560*lnTe-5.73932875*(lnTe**2)+1.56315498*(lnTe**3)-0.28770560*(lnTe**4)+3.48255977e-2*(lnTe**5)-2.63197617e-3*(lnTe**6)+1.11954395e-4*(lnTe**7)-2.03914985e-6*(lnTe**8))            
                rate = sp.exp(-32.71396786+13.5365560*lnTe-5.73932875*(lnTe**2)+1.56315498*(lnTe**3)-0.28770560*(lnTe**4)+3.48255977e-2*(lnTe**5)-2.63197617e-3*(lnTe**6)+1.11954395e-4*(lnTe**7)-2.03914985e-6*(lnTe**8))            
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hp').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and {ChemSpecie('h').sym_name} == set(self.products):
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                #reactions 2 and 3

                expr1 = 3.92e-13*invTe**0.6353
                expr2 = sp.exp(-28.61303380689232-0.7241125657826851*lnTe-0.02026044731984691*lnTe**2-0.002380861877349834*lnTe**3-0.0003212605213188796*lnTe**4-0.00001421502914054107*lnTe**5+4.989108920299513e-6*lnTe**6+5.755614137575758e-7*lnTe**7-1.856767039775261e-8*lnTe**8-3.071135243196595e-9*lnTe**9)
                
                rate = sp.Piecewise((expr1, Te <= 5.5e3), (expr2, Te > 5.5e3))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('he').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('hep').sym_name] == 1 and Counter(self.products)[ChemSpecie('elec').sym_name] == 2:
            #reaction 4
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = sp.exp(-44.09864886+23.91596563*lnTe-10.7532302*(lnTe**2)+3.05803875*(lnTe**3)-0.56851189*(lnTe**4)+6.79539123e-2*(lnTe**5)-5.00905610e-3*(lnTe**6)+2.06723616e-4*(lnTe**7)-3.64916141e-6*(lnTe**8))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and {ChemSpecie('he').sym_name} == set(self.products):
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                #reactions 5 and 6
                expr1 = 3.92e-13*invTe**0.6353
                expr2 = 1.54e-9*(1.0+0.30/sp.exp(8.099328789667*invTe))/(sp.exp(40.49664394833662*invTe)*Te**1.50)+3.92e-13/Te**0.6353

                rate = sp.Piecewise((expr1, Te <= 9.28e3), (expr2, Te > 9.28e3))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('hepp').sym_name] == 1 and Counter(self.products)[ChemSpecie('elec').sym_name] == 2:
            #reaction 7
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = sp.exp(-68.71040990212001+43.93347632635*lnTe-18.48066993568*lnTe**2+4.701626486759002*lnTe**3-0.7692466334492*lnTe**4+0.08113042097303*lnTe**5-0.005324020628287001*lnTe**6+0.0001975705312221*lnTe**7-3.165581065665e-6*lnTe**8)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hepp').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and {ChemSpecie('hep').sym_name} == set(self.products):
            #reaction 8
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1.891e-10/(sp.sqrt(Tgas/9.37)*(1.+sp.sqrt(Tgas/9.37))**0.2476*(1.+sp.sqrt(Tgas/2.774e6))**1.7524)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and {ChemSpecie('hm').sym_name} == set(self.products):
            #reaction 9
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1.4e-18*Tgas**0.928*sp.exp(-Tgas/16200.)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm').sym_name, ChemSpecie('h').sym_name} == set(self.reactants) and {ChemSpecie('h2').sym_name, ChemSpecie('elec').sym_name} == set(self.products):
            #reaction 10
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                a1=1.3500e-09
                a2=9.8493e-02
                a3=3.2852e-01
                a4=5.5610e-01
                a5=2.7710e-07
                a6=2.1826e+00
                a7=6.1910e-03
                a8=1.0461e+00
                a9=8.9712e-11
                a10=3.0424e+00
                a11=3.2576e-14
                a12=3.7741e+00
                rate = a1*(Tgas**a2+a3*Tgas**a4+a5*Tgas**a6)/(1.+a7*Tgas**a8+a9*Tgas**a10+a11*Tgas**a12)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h').sym_name, ChemSpecie('hp').sym_name} == set(self.reactants) and {ChemSpecie('h2p').sym_name} == set(self.products):
            #reactions 11 and 12
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                expr1 = 2.10e-20*(Tgas/30.)**(-0.15)
                expr2 = 10**(-18.20-3.194*sp.log(Tgas, 10)+1.786*(sp.log(Tgas, 10))**2-0.2072*(sp.log(Tgas, 10))**3)
                rate = sp.Piecewise((expr1, Tgas < 30), (expr2, Tgas >= 30))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p').sym_name, ChemSpecie('h').sym_name} == set(self.reactants) and {ChemSpecie('hp').sym_name, ChemSpecie('h2').sym_name} == set(self.products):
            #reaction 13
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 6.0e-10
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2').sym_name, ChemSpecie('hp').sym_name} == set(self.reactants) and {ChemSpecie('h2p').sym_name, ChemSpecie('h').sym_name} == set(self.products):
            #reaction 14
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):

                asav = 2.1237150e4
                bsav1=-3.3232183e-7
                bsav2= 3.3735382e-7
                bsav3=-1.4491368e-7 
                bsav4= 3.4172805e-8 
                bsav5=-4.7813728e-9 
                bsav6= 3.9731542e-10
                bsav7=-1.8171411e-11
                bsav8= 3.5311932e-13
                sumsav=bsav1+bsav2*sp.log(Tgas)+bsav3*(sp.log(Tgas))**2+bsav4*(sp.log(Tgas))**3+bsav5*(sp.log(Tgas))**4+bsav6*(sp.log(Tgas))**5+bsav7*(sp.log(Tgas))**6+bsav8*(sp.log(Tgas))**7
                expr = sumsav*sp.exp(-asav*invT)

                rate = sp.Piecewise((expr, sp.And(Tgas >= 1e2, Tgas <= 3e4)), (0, True))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and {ChemSpecie('h').sym_name, ChemSpecie('hm').sym_name} == set(self.products):
            #reaction 15
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 3.55e1*Tgas**(-2.28)*sp.exp(-46707./Tgas)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('h').sym_name] == 2 and Counter(self.products)[ChemSpecie('elec').sym_name] == 1:
            #reaction 16
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 4.38e-10*sp.exp(-102000./Tgas)*Tgas**(0.35)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2').sym_name, ChemSpecie('h').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('h').sym_name] == 3:
            #reaction 17
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):


                k_CIDm = np.zeros((2, 21))
                k_CIDm[0] = (-178.4239, -68.42243, 43.20243, -4.633167, \
                             69.70086, 40870.38, -23705.70, 128.8953, -53.91334, \
                             5.315517, -19.73427, 16780.95, -25786.11, 14.82123, \
                             -4.890915, 0.4749030, -133.8283, -1.164408, 0.8227443, \
                             0.5864073, -2.056313)

                k_CIDm[1] = (-142.7664, 42.70741, -2.027365, -0.2582097, \
                              21.36094, 27535.31, -21467.79, 60.34928, -27.43096, \
                              2.676150, -11.28215, 14254.55, -23125.20, 9.305564, \
                             -2.464009, 0.1985955, 743.0600, -1.174242, 0.7502286, \
                             0.2358848, 2.937507)

                n_H  = self.get_Hnuclei(composition, density)
                logT = sp.log(Tgas, 10)
                invT = 1.0/Tgas
                logT2 = logT*logT
                logT3 = logT2*logT
                logTv = np.array([1.0, logT, logT2, logT3])
                k_CID = 0.

                i = 0
                while i < 2:
                    logk_h1 = k_CIDm[i,0]*logTv[0] + k_CIDm[i,1]*logTv[1] + \
                              k_CIDm[i,2]*logTv[2] + k_CIDm[i,3]*logTv[3] + \
                              k_CIDm[i,4]*sp.log(1.0+k_CIDm[i,5]*invT, 10)

                    logk_h2 = k_CIDm[i,6]*invT

                    logk_l1 = k_CIDm[i,7]*logTv[0] + k_CIDm[i,8]*logTv[1] + \
                              k_CIDm[i,9]*logTv[2] + k_CIDm[i,10]*sp.log(1.0+k_CIDm[i,11]*invT, 10)

                    logk_l2 = k_CIDm[i,12]*invT

                    logn_c1 = k_CIDm[i,13]*logTv[0] + k_CIDm[i,14]*logTv[1] + \
                              k_CIDm[i,15]*logTv[2] + k_CIDm[i,16]*invT

                    logn_c2 = k_CIDm[i,17] + logn_c1

                    p = k_CIDm[i,18] + k_CIDm[i,19]*sp.exp(-Tgas/1.850e3) + \
                        k_CIDm[i,20]*sp.exp(-Tgas/4.40e2)

                    n_c1 = 1e1**(logn_c1)
                    n_c2 = 1e1**(logn_c2)

                    logk_CID = logk_h1 - (logk_h1 - logk_l1) / (1.0 + (n_H/n_c1)**p) + \
                               logk_h2 - (logk_h2 - logk_l2) / (1.0 + (n_H/n_c2)**p)

                    k_CID = k_CID + 1.e1**logk_CID

                    i += 1

                return k_CID

                #rate = 0. #self.dissH2_Martin96(Tgas, composition)
                #return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('h').sym_name] == 1 and Counter(self.products)[ChemSpecie('elec').sym_name] == 2:
            #reaction 18
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = sp.exp(-18.01849334273+2.360852208681*lnTe-0.2827443061704*lnTe**2+0.01623316639567*lnTe**3-0.03365012031362999*lnTe**4+0.01178329782711*lnTe**5-0.001656194699504*lnTe**6+0.0001068275202678*lnTe**7-2.631285809207e-6*lnTe**8)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm').sym_name, ChemSpecie('h').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('h').sym_name] == 2 and Counter(self.products)[ChemSpecie('elec').sym_name] == 1:
            #reactions 19 and 20
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                expr1 = 2.56e-9*Te**1.78186
                expr2 = sp.exp(-20.37260896533324+1.139449335841631*lnTe-0.1421013521554148*lnTe**2+0.00846445538663*lnTe**3-0.0014327641212992*lnTe**4+0.0002012250284791*lnTe**5+0.0000866396324309*lnTe**6-0.00002585009680264*lnTe**7+2.4555011970392e-6*lnTe**8-8.06838246118e-8*lnTe**9)
                rate = sp.Piecewise((expr1, Tgas <= 1.16e3), (expr2, Tgas > 1.16e3))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm').sym_name, ChemSpecie('hp').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('h').sym_name] == 2:
            #reaction 21
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                expr = (2.96e-6/sp.sqrt(Tgas)-1.73e-9+2.50e-10*sp.sqrt(Tgas)-7.77e-13*Tgas)
                rate = sp.Piecewise((expr, sp.And(Tgas >= 1e1, Tgas <= 1e5)), (0, True))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm').sym_name, ChemSpecie('hp').sym_name} == set(self.reactants) and {ChemSpecie('h2p').sym_name, ChemSpecie('elec').sym_name} == set(self.products):
            #reaction 22
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1e-8*Tgas**(-0.4)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('h').sym_name] == 2:
            #reaction 23
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                expr = 1e6*(4.2278e-14-2.3088e-17*Tgas+7.3428e-21*Tgas**2-7.5474e-25*Tgas**3+3.3468e-29*Tgas**4-5.528e-34*Tgas**5)
                rate = sp.Piecewise((expr, Tgas <= 1e4), (0, Tgas > 1e4))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p').sym_name, ChemSpecie('hm').sym_name} == set(self.reactants) and {ChemSpecie('h').sym_name, ChemSpecie('h2').sym_name} == set(self.products):
            #reaction 24
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 5e-7*sp.sqrt(1.e2*invT)
                return rate
            self.rate_function = rate_function

        elif Counter(self.reactants)[ChemSpecie('h').sym_name] == 3 and {ChemSpecie('h2').sym_name, ChemSpecie('h').sym_name} == set(self.products):
            #reaction 25
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 6e-32*Tgas**(-0.25)+2e-31*Tgas**(-0.5)
                return rate
            self.rate_function = rate_function

        elif Counter(self.reactants)[ChemSpecie('h').sym_name] == 2 and Counter(self.reactants)[ChemSpecie('h2').sym_name] == 1 and Counter(self.products)[ChemSpecie('h2').sym_name] == 2:
            #reaction 26
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = (6e-32*Tgas**(-0.25)+2e-31*Tgas**(-0.5))/8.0
                return rate
            self.rate_function = rate_function

        elif Counter(self.reactants)[ChemSpecie('h2').sym_name] == 2 and Counter(self.products)[ChemSpecie('h2').sym_name] == 1 and Counter(self.products)[ChemSpecie('h').sym_name] == 2:
            #reaction 27
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                Hnuclei = self.get_Hnuclei(composition, density)
                kl21 = 1.18e-10*sp.exp(-6.95e4*invT)
                kh21 = 8.125e-8*Tgas**(-0.5)*sp.exp(-5.2e4*invT)*(1.0-sp.exp(-6e3*invT))
                ncr21 = 1e1**(4.845-1.3*sp.log(Tgas*1e-4, 10)+1.62*(sp.log(Tgas*1e-4, 10))**2)
                a21=1.0/(1.0+(Hnuclei/ncr21))
                rate = kh21**(1.0 - a21)*kl21**a21
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep').sym_name, ChemSpecie('h').sym_name} == set(self.reactants) and {ChemSpecie('he').sym_name, ChemSpecie('hp').sym_name} == set(self.products):
            #reaction 28
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1.20e-15*(Tgas/3e2)**0.25
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('he').sym_name, ChemSpecie('hp').sym_name} == set(self.reactants) and {ChemSpecie('hep').sym_name, ChemSpecie('h').sym_name} == set(self.products):
            #reactions 29 and 30
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                expr1 = 1.26e-9*Tgas**(-0.75)*sp.exp(-1.275e5*invT)
                expr2 = 4e-37*Tgas**(4.74)
                rate = sp.Piecewise((expr1, Tgas <= 1e4), (expr2, Tgas > 1e4))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2').sym_name, ChemSpecie('dp').sym_name} == set(self.reactants) and {ChemSpecie('hd').sym_name, ChemSpecie('hp').sym_name} == set(self.products):
            #reaction 31
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1e-9*(0.417+0.846*sp.log(Tgas, 10)-0.137*(sp.log(Tgas, 10))**2)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hd').sym_name, ChemSpecie('hp').sym_name} == set(self.reactants) and {ChemSpecie('h2').sym_name, ChemSpecie('dp').sym_name} == set(self.products):
            #reaction 32
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1e-9*sp.exp(-4.57e2*invT)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2').sym_name, ChemSpecie('d').sym_name} == set(self.reactants) and {ChemSpecie('hd').sym_name, ChemSpecie('h').sym_name} == set(self.products):
            #reactions 33 and 34
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                expr1 = 10**(-56.4737+5.88886*sp.log(Tgas, 10)+7.19692*(sp.log(Tgas, 10))**2+2.25069*(sp.log(Tgas, 10))**3-2.16903*(sp.log(Tgas, 10))**4+0.317887*(sp.log(Tgas, 10))**5)
                expr2 = 3.17e-10*sp.exp(-5207.*invT)
                rate = sp.Piecewise((expr1, Tgas <= 1.167479642374226e3), (expr2, Tgas > 1.167479642374226e3))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hd').sym_name, ChemSpecie('h').sym_name} == set(self.reactants) and {ChemSpecie('h2').sym_name, ChemSpecie('d').sym_name} == set(self.products):
            #reaction 35
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                expr = 5.25e-11*sp.exp(-4430.*invT+1.739e5*(invT)**2)
                rate = sp.Piecewise((expr, Tgas > 2e2), (0, Tgas <= 2e2))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('d').sym_name, ChemSpecie('hm').sym_name} == set(self.reactants) and {ChemSpecie('hd').sym_name, ChemSpecie('elec').sym_name} == set(self.products):
            #reaction 36
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1.5e-9*(T32)**(-0.1)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hp').sym_name, ChemSpecie('d').sym_name} == set(self.reactants) and {ChemSpecie('h').sym_name, ChemSpecie('dp').sym_name} == set(self.products):
            #reaction 37
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                expr = 2e-10*Tgas**(0.402)*sp.exp(-37.1*invT)-3.31e-17*Tgas**(1.48)
                rate = sp.Piecewise((expr, Tgas >= 5e1), (0, Tgas < 5e1))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h').sym_name, ChemSpecie('dp').sym_name} == set(self.reactants) and {ChemSpecie('hp').sym_name, ChemSpecie('d').sym_name} == set(self.products):
            #reaction 38
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                expr = 2.06e-10*Tgas**(0.396)*sp.exp(-33.0*invT)+2.03e-9*Tgas**(-0.332)
                rate = sp.Piecewise((expr, Tgas >= 5e1), (0, Tgas < 5e1))
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('dp').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and {ChemSpecie('d').sym_name} == set(self.products):
            #reaction 39
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 3.6e-12*(Tgas/300)**(-0.75)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h').sym_name, ChemSpecie('d').sym_name} == set(self.reactants) and {ChemSpecie('hd').sym_name} == set(self.products):
            #reaction 40
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1e-25
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hdp').sym_name, ChemSpecie('h').sym_name} == set(self.reactants) and {ChemSpecie('hd').sym_name, ChemSpecie('hp').sym_name} == set(self.products):
            #reaction 41
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 6.4e-10
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hp').sym_name, ChemSpecie('d').sym_name} == set(self.reactants) and {ChemSpecie('hdp').sym_name} == set(self.products):
            #reaction 42
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 10.0**(-19.38-1.523*sp.log(Tgas, 10)+1.118*(sp.log(Tgas, 10))**2.0-0.1269*(sp.log(Tgas, 10))**3.0)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h').sym_name, ChemSpecie('dp').sym_name} == set(self.reactants) and {ChemSpecie('hdp').sym_name} == set(self.products):
            #reaction 43
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 10.0**(-19.38-1.523*sp.log(Tgas, 10)+1.118*(sp.log(Tgas, 10))**2.0-0.1269*(sp.log(Tgas, 10))**3.0)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hdp').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and {ChemSpecie('h').sym_name, ChemSpecie('d').sym_name} == set(self.products):
            #reaction 44
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                invsqrT = 1.0/sp.sqrt(Tgas)
                rate = 7.2e-8*invsqrT
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('d').sym_name, ChemSpecie('elec').sym_name} == set(self.reactants) and {ChemSpecie('dm').sym_name} == set(self.products):
            #reaction 45
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 3e-16*(Tgas/300)**(0.95)*sp.exp(-Tgas/9.320e3)
                return rate
            self.rate_function = rate_function
            
        elif {ChemSpecie('dp').sym_name, ChemSpecie('dm').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('d').sym_name] == 2:
            #reaction 46
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 5.7e-8*(Tgas/300)**(-0.5)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('dm').sym_name, ChemSpecie('hp').sym_name} == set(self.reactants) and {ChemSpecie('h').sym_name, ChemSpecie('d').sym_name} == set(self.products):
            #reaction 47
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 4.6e-8*(Tgas/300)**(-0.5)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hm').sym_name, ChemSpecie('d').sym_name} == set(self.reactants) and {ChemSpecie('dm').sym_name, ChemSpecie('h').sym_name} == set(self.products):
            #reaction 48
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 6.4e-9*(Tgas/300)**(0.41)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('dm').sym_name, ChemSpecie('h').sym_name} == set(self.reactants) and {ChemSpecie('hm').sym_name, ChemSpecie('d').sym_name} == set(self.products):
            #reaction 49
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 6.4e-9*(Tgas/300)**(0.41)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('dm').sym_name, ChemSpecie('h').sym_name} == set(self.reactants) and {ChemSpecie('hd').sym_name, ChemSpecie('elec').sym_name} == set(self.products):
            #reaction 50
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1.5e-9*(Tgas/300)**(-0.1)
                return rate
            self.rate_function = rate_function

        #new reactions, added by Piyush Sharda from SLD98
        elif {ChemSpecie('dp').sym_name, ChemSpecie('hm').sym_name} == set(self.reactants) and {ChemSpecie('d').sym_name, ChemSpecie('h').sym_name} == set(self.products):
            #reaction 51
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 4.6e-8*(Tgas/300)**(-0.5)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep').sym_name, ChemSpecie('hm').sym_name} == set(self.reactants) and {ChemSpecie('he').sym_name, ChemSpecie('h').sym_name} == set(self.products):
            #reaction 52
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 2.32e-7*((Tgas/300)**(-0.52))*sp.exp(Tgas/22400)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep').sym_name, ChemSpecie('d').sym_name} == set(self.reactants) and {ChemSpecie('he').sym_name, ChemSpecie('d+').sym_name} == set(self.products):
            #reaction 53
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1.1e-15*(Tgas/300)**(0.25)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep').sym_name, ChemSpecie('dm').sym_name} == set(self.reactants) and {ChemSpecie('he').sym_name, ChemSpecie('d').sym_name} == set(self.products):
            #reaction 54
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 3.03e-7*((Tgas/300)**(-0.52))*sp.exp(Tgas/22400)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2').sym_name, ChemSpecie('hep').sym_name} == set(self.reactants) and {ChemSpecie('h2p').sym_name, ChemSpecie('he').sym_name} == set(self.products):
            #reaction 55
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 7.2e-15
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p').sym_name, ChemSpecie('d').sym_name} == set(self.reactants) and {ChemSpecie('hdp').sym_name, ChemSpecie('h').sym_name} == set(self.products):
            #reaction 56
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1.07e-9*((Tgas/300)**(6.2e-2))*sp.exp(Tgas/41400)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p').sym_name, ChemSpecie('d').sym_name} == set(self.reactants) and {ChemSpecie('h2').sym_name, ChemSpecie('dp').sym_name} == set(self.products):
            #reaction 57
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 6.4e-10
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hdp').sym_name, ChemSpecie('h').sym_name} == set(self.reactants) and {ChemSpecie('h2p').sym_name, ChemSpecie('d').sym_name} == set(self.products):
            #reaction 58
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1.0e-9*sp.exp(154/Tgas)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2').sym_name, ChemSpecie('hep').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('he').sym_name] == 1 and Counter(self.products)[ChemSpecie('h').sym_name] == 2:
            #reaction 59
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 3.7e-14*sp.exp(-35/Tgas)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('h2p').sym_name, ChemSpecie('hm').sym_name} == set(self.reactants) and Counter(self.products)[ChemSpecie('h').sym_name] == 3:
            #reaction 60
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 1.4e-7*(Tgas/300)**(-0.5)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep').sym_name, ChemSpecie('hd').sym_name} == set(self.reactants) and {ChemSpecie('he').sym_name, ChemSpecie('hp').sym_name, ChemSpecie('d').sym_name} == set(self.products):
            #reaction 61
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 5.5e-14*(Tgas/300)**(-0.24)
                return rate
            self.rate_function = rate_function

        elif {ChemSpecie('hep').sym_name, ChemSpecie('hd').sym_name} == set(self.reactants) and {ChemSpecie('he').sym_name, ChemSpecie('h').sym_name, ChemSpecie('dp').sym_name} == set(self.products):
            #reaction 62
            def rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density = 0):
                rate = 5.5e-14*(Tgas/300)**(-0.24)
                return rate
            self.rate_function = rate_function

        else:
            raise UnsupportedSympyChemRate()


    def get_Hnuclei(self, composition, density=0):
        import sympy as sp

        if self.massfracs == 1:
            if density == 0:
                raise ValueError('If working in mass fractions, you must also input the density in g/cm^3 ')

            nH = composition[ChemSpecie('hp').sym_name]/ChemSpecie('hp').m + composition[ChemSpecie('h').sym_name]/ChemSpecie('h').m + composition[ChemSpecie('hm').sym_name]/ChemSpecie('hm').m + \
                 composition[ChemSpecie('h2').sym_name]*2.0/ChemSpecie('h2').m + composition[ChemSpecie('h2p').sym_name]*2.0/ChemSpecie('h2p').m

            if ChemSpecie('hd').sym_name in composition:
                nH += composition[ChemSpecie('hd').sym_name]/ChemSpecie('hd').m + composition[ChemSpecie('hdp').sym_name]/ChemSpecie('hdp').m

            return nH*density

        elif self.massfracs == 0:
            nH = composition[ChemSpecie('hp').sym_name] + composition[ChemSpecie('h').sym_name] + composition[ChemSpecie('hm').sym_name] + \
                 composition[ChemSpecie('h2').sym_name]*2.0 + composition[ChemSpecie('h2p').sym_name]*2.0

            if ChemSpecie('hd').sym_name in composition:
                nH += composition[ChemSpecie('hd').sym_name] + composition[ChemSpecie('hdp').sym_name]

            return nH

    def get_small(self, composition, density=0):
        import sympy as sp

        if self.massfracs == 1:
            nmax = sp.Max(composition[ChemSpecie('elec').sym_name]/ChemSpecie('elec').m, composition[ChemSpecie('hp').sym_name]/ChemSpecie('hp').m, \
                          composition[ChemSpecie('h').sym_name]/ChemSpecie('h').m, composition[ChemSpecie('hm').sym_name]/ChemSpecie('hm').m, \
                          composition[ChemSpecie('dp').sym_name]/ChemSpecie('dp').m, composition[ChemSpecie('d').sym_name]/ChemSpecie('d').m, \
                          composition[ChemSpecie('h2p').sym_name]/ChemSpecie('h2p').m, composition[ChemSpecie('dm').sym_name]/ChemSpecie('dm').m, \
                          composition[ChemSpecie('h2').sym_name]/ChemSpecie('h2').m, composition[ChemSpecie('hdp').sym_name]/ChemSpecie('hdp').m, \
                          composition[ChemSpecie('hd').sym_name]/ChemSpecie('hd').m, composition[ChemSpecie('hepp').sym_name]/ChemSpecie('hepp').m, \
                          composition[ChemSpecie('hep').sym_name]/ChemSpecie('hep').m, composition[ChemSpecie('he').sym_name]/ChemSpecie('he').m)
            nmaxx = nmax*density
            small = 1e-40/(nmax**3)
            return small

        elif self.massfracs == 0:
            print('Case not yet implemented!')
            return None

    def eval(self, T, composition, density=0):
        import sympy as sp
        Tgas = T
        Te = Tgas*8.617343e-5 #CHECK KROME FILES!!!
        invTe = 1.0/Te
        invT = 1.0/T
        lnTe = sp.log(Te)
        T32 = Tgas*0.0033333333333333335 #Tgas/(300 K)

        return self.rate_function(Tgas, Te, invTe, invT, lnTe, T32, composition, density)

    def __repr__(self):
        repstring = str(self.reactants) + ' --> ' + str(self.products)
        return repstring

class UnsupportedChemRate(BaseException):
    def __init__(self):
        print('The chemical rate for the specie(s) you entered is unsupported')
        return

class UnsupportedSympyChemRate(BaseException):
    def __init__(self):
        print('The sympy chemical rate for the specie(s) you entered is unsupported')
        return

class UnphysicalTemperature(BaseException):
    def __init__(self):
        print('The temperature you entered is <= 0 K')
        return