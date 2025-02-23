# unit tests for rates
import pynucastro.networks as networks
import pynucastro.rates as rates
import os
import filecmp
import glob

import io


class TestStarKillerCxxNetwork:
    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """
        pass

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """
        pass

    def setup_method(self):
        """ this is run before each test """
        files = ["c12-c12a-ne20-cf88",
                 "c12-c12n-mg23-cf88",
                 "c12-c12p-na23-cf88",
                 "c12-ag-o16-nac2",
                 "na23--ne23-toki",
                 "ne23--na23-toki",
                 "n--p-wc12"]

        self.fn = networks.StarKillerCxxNetwork(files)
        self.fn.secret_code = "testing"

    def teardown_method(self):
        """ this is run after each test """
        self.tf = None

    def cromulent_ftag(self, ftag, answer, n_indent=1):
        """ check to see if function ftag returns answer """

        output = io.StringIO()
        ftag(n_indent, output)
        result = output.getvalue() == answer
        output.close()
        return result

    def test_nrat_reaclib(self):
        """ test the _nrat_reaclib function """

        answer = ('    const int NrateReaclib = 5;\n' +
                  '    const int NumReaclibSets = 6;\n')
        assert self.cromulent_ftag(self.fn._nrat_reaclib, answer, n_indent=1)

    def test_nrat_tabular(self):
        """ test the _nrat_tabular function """

        answer = '    const int NrateTabular = 2;\n'
        assert self.cromulent_ftag(self.fn._nrat_tabular, answer, n_indent=1)

    def test_nrxn(self):
        """ test the _nrxn function """

        answer = ('    k_c12_c12__he4_ne20 = 1,\n' +
                  '    k_c12_c12__n_mg23 = 2,\n' +
                  '    k_c12_c12__p_na23 = 3,\n' +
                  '    k_he4_c12__o16 = 4,\n' +
                  '    k_n__p__weak__wc12 = 5,\n' +
                  '    k_na23__ne23 = 6,\n' +
                  '    k_ne23__na23 = 7,\n' +
                  '    NumRates = k_ne23__na23\n')
        assert self.cromulent_ftag(self.fn._nrxn, answer, n_indent=1)

    def test_ebind(self):
        """ test the _ebind function """

        answer = ('        ebind_per_nucleon(N) = 0.0_rt;\n' +
                  '        ebind_per_nucleon(H1) = 0.0_rt;\n' +
                  '        ebind_per_nucleon(He4) = 7.073915_rt;\n' +
                  '        ebind_per_nucleon(C12) = 7.680144_rt;\n' +
                  '        ebind_per_nucleon(O16) = 7.976206_rt;\n' +
                  '        ebind_per_nucleon(Ne20) = 8.03224_rt;\n' +
                  '        ebind_per_nucleon(Ne23) = 7.955256_rt;\n' +
                  '        ebind_per_nucleon(Na23) = 8.111493000000001_rt;\n' +
                  '        ebind_per_nucleon(Mg23) = 7.901115_rt;\n')
        assert self.cromulent_ftag(self.fn._ebind, answer, n_indent=2)

    def test_write_network(self):
        """ test the write_network function"""
        test_path = "_test_cxx/"
        reference_path = "_starkiller_cxx_reference/"
        base_path = os.path.relpath(os.path.dirname(__file__))

        self.fn.write_network(odir=test_path)

        files = ["actual_network_data.cpp",
                 "actual_network.H",
                 "actual_rhs.H",
                 "inputs.burn_cell.VODE",
                 "Make.package",
                 "_parameters",
                 "reaclib_rate_metadata.dat",
                 "reaclib_rates.H",
                 "table_rates_data.cpp",
                 "table_rates.H"]

        errors = []
        for test_file in files:
            # note, _test is written under whatever directory pytest is run from,
            # so it is not necessarily at the same place as _starkiller_reference
            if not filecmp.cmp(os.path.normpath(f"{test_path}/{test_file}"),
                               os.path.normpath(f"{base_path}/{reference_path}/{test_file}"),
                               shallow=False):
                errors.append(test_file)

        assert not errors, f"errors: {' '.join(errors)}"
