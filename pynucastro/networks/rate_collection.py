"""A collection of classes and methods to deal with collections of
rates that together make up a network."""

# Common Imports
import warnings
import functools
import math
import os

from operator import mul, add
from collections import OrderedDict, Counter

from ipywidgets import interact

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import networkx as nx

# Import Rate
from pynucastro.rates import Rate, Nucleus, Library, ChemSpecie, ChemRate, ChemComposition, SympyChemRate

import sys
sys.path.append(os.path.abspath("/scratch/jh2/ps3459/pynucastro/pynucastro/rates"))
import constants as cons

mpl.rcParams['figure.dpi'] = 100

class Composition:
    """a composition holds the mass fractions of the nuclei in a network
    -- useful for evaluating the rates

    """
    def __init__(self, nuclei, small=1.e-16):
        """nuclei is an iterable of the nuclei (Nucleus objects) in the network"""
        if not isinstance(nuclei[0], Nucleus):
            raise ValueError("must supply an iterable of Nucleus objects")
        self.X = {k: small for k in nuclei}



    def set_solar_like(self, Z=0.02):
        """ approximate a solar abundance, setting p to 0.7, He4 to 0.3 - Z and
        the remainder evenly distributed with Z """
        num = len(self.X)
        rem = Z/(num-2)
        for k in self.X:
            if k == Nucleus("p"):
                self.X[k] = 0.7
            elif k.raw == "he4":
                self.X[k] = 0.3 - Z
            else:
                self.X[k] = rem

        self.normalize()

    def set_all(self, xval):
        """ set all species to a particular value """
        for k in self.X:
            self.X[k] = xval

    def set_nuc(self, name, xval):
        """ set nuclei name to the mass fraction xval """
        for k in self.X:
            if k.raw == name:
                self.X[k] = xval
                break

    def normalize(self):
        """ normalize the mass fractions to sum to 1 """
        X_sum = sum(self.X[k] for k in self.X)

        for k in self.X:
            self.X[k] /= X_sum

    def get_molar(self):
        """ return a dictionary of molar fractions"""
        molar_frac = {k: v/k.A for k, v in self.X.items()}
        return molar_frac

    def eval_ye(self):
        """ return the electron fraction """
        zvec = []
        avec = []
        xvec = []
        for n in self.X:
            zvec.append(n.Z)
            avec.append(n.A)
            xvec.append(self.X[n])
        zvec = np.array(zvec)
        avec = np.array(avec)
        xvec = np.array(xvec)
        electron_frac = np.sum(zvec*xvec/avec)/np.sum(xvec)
        return electron_frac

    def __str__(self):
        ostr = ""
        for k in self.X:
            ostr += f"  X({k}) : {self.X[k]}\n"
        return ostr

class ScreeningPair:
    """a pair of nuclei that will have rate screening applied.  We store a
    list of all rates that match this pair of nuclei"""

    def __init__(self, name, nuc1, nuc2, rate=None):
        self.name = name
        self.n1 = nuc1
        self.n2 = nuc2

        if rate is None:
            self.rates = []
        else:
            self.rates = [rate]

    def add_rate(self, rate):
        self.rates.append(rate)

    def __eq__(self, other):
        """all we care about is whether the names are the same -- that conveys
        what the reaction is"""

        return self.name == other.name

class RateCollection:
    """ a collection of rates that together define a network """

    pynucastro_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    def __init__(self, rate_files=None, libraries=None, rates=None, precedence=(),
                 symmetric_screening=False, do_screening=True):
        """rate_files are the files that together define the network.  This
        can be any iterable or single string.

        This can include Reaclib library files storing multiple rates.

        If libraries is supplied, initialize a RateCollection using the rates
        in the Library object(s) in list 'libraries'.

        If rates is supplied, initialize a RateCollection using the
        Rate objects in the list 'rates'.

        Precedence should be sequence of rate labels (e.g. wc17) to be used to
        resolve name conflicts. If a nonempty sequence is provided, the rate
        collection will automatically be scanned for multiple rates with the
        same name. If all of their labels were given a ranking, the rate with
        the label that comes first in the sequence will be retained and the
        rest discarded.

        symmetric_screening means that we screen the reverse rates
        using the same factor as the forward rates, for rates computed
        via detailed balance.

        Any combination of these options may be supplied.

        """

        self.files = []
        self.rates = []
        self.library = None

        self.symmetric_screening = symmetric_screening
        self.do_screening = do_screening

        if rate_files:
            if isinstance(rate_files, str):
                rate_files = [rate_files]
            self._read_rate_files(rate_files)

        if rates:
            if isinstance(rates, Rate):
                rates = [rates]
            try:
                for r in rates:
                    assert isinstance(r, Rate)
            except:
                print('Expected Rate object or list of Rate objects passed as the rates argument.')
                raise
            else:
                rlib = Library(rates=rates)
                if not self.library:
                    self.library = rlib
                else:
                    self.library = self.library + rlib

        if libraries:
            if isinstance(libraries, Library):
                libraries = [libraries]
            try:
                for lib in libraries:
                    assert isinstance(lib, Library)
            except:
                print('Expected Library object or list of Library objects passed as the libraries argument.')
                raise
            else:
                if not self.library:
                    self.library = libraries.pop(0)
                for lib in libraries:
                    self.library = self.library + lib

        if self.library:
            self.rates = self.rates + self.library.get_rates()

        if precedence:
            self._make_distinguishable(precedence)

        # get the unique nuclei
        u = []
        print(self.rates)
        for r in self.rates:
            t = set(r.reactants + r.products)
            u = set(list(u) + list(t))

        self.unique_nuclei = sorted(u)

        # now make a list of each rate that touches each nucleus
        # we'll store this in a dictionary keyed on the nucleus
        self.nuclei_consumed = OrderedDict()
        self.nuclei_produced = OrderedDict()

        for n in self.unique_nuclei:
            self.nuclei_consumed[n] = [r for r in self.rates if n in r.reactants]
            self.nuclei_produced[n] = [r for r in self.rates if n in r.products]

        # Re-order self.rates so Reaclib rates come first,
        # followed by Tabular rates. This is needed if
        # reaclib coefficients are targets of a pointer array
        # in the Fortran network.
        # It is desired to avoid wasting array size
        # storing meaningless Tabular coefficient pointers.
        self.rates = sorted(self.rates,
                            key=lambda r: r.chapter == 't')

        self.tabular_rates = []
        self.reaclib_rates = []
        for n, r in enumerate(self.rates):
            if r.chapter == 't':
                self.tabular_rates.append(n)
            elif isinstance(r.chapter, int):
                self.reaclib_rates.append(n)
            else:
                print('ERROR: Chapter type unknown for rate chapter {}'.format(
                    str(r.chapter)))
                exit()

    def _read_rate_files(self, rate_files):
        # get the rates
        self.files = rate_files
        for rf in self.files:
            try:
                rflib = Library(rf)
            except:
                print(f"Error reading library from file: {rf}")
                raise
            else:
                if not self.library:
                    self.library = rflib
                else:
                    self.library = self.library + rflib

    def get_nuclei(self):
        """ get all the nuclei that are part of the network """
        return self.unique_nuclei

    def evaluate_rates(self, rho, T, composition):
        """evaluate the rates for a specific density, temperature, and
        composition"""
        rvals = OrderedDict()
        ys = composition.get_molar()
        y_e = composition.eval_ye()

        for r in self.rates:
            val = r.prefactor * rho**r.dens_exp * r.eval(T, rho * y_e)
            if (r.weak_type == 'electron_capture' and not r.tabular):
                val = val * y_e
            yfac = functools.reduce(mul, [ys[q] for q in r.reactants])
            rvals[r] = yfac * val

        return rvals

    def evaluate_ydots(self, rho, T, composition):
        """evaluate net rate of change of molar abundance for each nucleus
        for a specific density, temperature, and composition"""

        rvals = self.evaluate_rates(rho, T, composition)
        ydots = dict()

        for nuc in self.unique_nuclei:

            # Rates that consume / produce nuc
            consuming_rates = self.nuclei_consumed[nuc]
            producing_rates = self.nuclei_produced[nuc]
            # Number of nuclei consumed / produced
            nconsumed = (r.reactants.count(nuc) for r in consuming_rates)
            nproduced = (r.products.count(nuc) for r in producing_rates)
            # Multiply each rate by the count
            consumed = (c * rvals[r] for c, r in zip(nconsumed, consuming_rates))
            produced = (c * rvals[r] for c, r in zip(nproduced, producing_rates))
            # Net change is difference between produced and consumed
            ydots[nuc] = sum(produced) - sum(consumed)

        return ydots

    def evaluate_activity(self, rho, T, composition):
        """sum over all of the terms contributing to ydot,
        neglecting sign"""

        rvals = self.evaluate_rates(rho, T, composition)
        act = dict()

        for nuc in self.unique_nuclei:

            # Rates that consume / produce nuc
            consuming_rates = self.nuclei_consumed[nuc]
            producing_rates = self.nuclei_produced[nuc]
            # Number of nuclei consumed / produced
            nconsumed = (r.reactants.count(nuc) for r in consuming_rates)
            nproduced = (r.products.count(nuc) for r in producing_rates)
            # Multiply each rate by the count
            consumed = (c * rvals[r] for c, r in zip(nconsumed, consuming_rates))
            produced = (c * rvals[r] for c, r in zip(nproduced, producing_rates))
            # Net activity is sum of produced and consumed
            act[nuc] = sum(produced) + sum(consumed)

        return act

    def network_overview(self):
        """ return a verbose network overview """
        ostr = ""
        for n in self.unique_nuclei:
            ostr += f"{n}\n"
            ostr += "  consumed by:\n"
            for r in self.nuclei_consumed[n]:
                ostr += f"     {r.string}\n"

            ostr += "  produced by:\n"
            for r in self.nuclei_produced[n]:
                ostr += f"     {r.string}\n"

            ostr += "\n"
        return ostr

    def get_screening_map(self):
        """a screening map is just a list of tuples containing the information
        about nuclei pairs for screening: (descriptive name of nuclei,
        nucleus 1, nucleus 2, rate, 1-based index of rate).  If symmetric_screening=True,
        then for reverse rates, we screen using the forward rate nuclei (assuming that we
        got here via detailed balance).

        """
        screening_map = []
        if not self.do_screening:
            return screening_map

        for r in self.rates:
            screen_nuclei = r.ion_screen
            if self.symmetric_screening:
                screen_nuclei = r.symmetric_screen

            if screen_nuclei:
                nucs = "_".join([str(q) for q in screen_nuclei])
                in_map = False

                scr = [q for q in screening_map if q.name == nucs]

                assert len(scr) <= 1

                if scr:
                    # we already have the reactants in our map, so we
                    # will already be doing the screening factors.
                    # Just append this new rate to the list we are
                    # keeping of the rates where this screening is
                    # needed

                    scr[0].add_rate(r)

                    # if we got here because nuc == "he4_he4_he4",
                    # then we also have to add to "he4_he4_he4_dummy"

                    if nucs == "he4_he4_he4":
                        scr2 = [q for q in screening_map if q.name == nucs + "_dummy"]
                        assert len(scr2) == 1

                        scr2[0].add_rate(r)

                else:

                    # we handle 3-alpha specially -- we actually need
                    # 2 screening factors for it

                    if nucs == "he4_he4_he4":
                        # he4 + he4
                        scr1 = ScreeningPair(nucs, screen_nuclei[0], screen_nuclei[1], r)

                        # he4 + be8
                        be8 = Nucleus("Be8", dummy=True)
                        scr2 = ScreeningPair(nucs + "_dummy", screen_nuclei[2], be8, r)

                        screening_map.append(scr1)
                        screening_map.append(scr2)

                    else:
                        scr1 = ScreeningPair(nucs, screen_nuclei[0], screen_nuclei[1], r)
                        screening_map.append(scr1)

        return screening_map

    def write_network(self, *args, **kwargs):
        """Before writing the network, check to make sure the rates
        are distinguishable by name."""
        assert self._distinguishable_rates(), "ERROR: Rates not uniquely identified by Rate.fname"
        self._write_network(*args, **kwargs)

    def _distinguishable_rates(self):
        """Every Rate in this RateCollection should have a unique Rate.fname,
        as the network writers distinguish the rates on this basis."""
        names = [r.fname for r in self.rates]
        for n, r in zip(names, self.rates):
            k = names.count(n)
            if k > 1:
                print(f'Found rate {r} named {n} with {k} entries in the RateCollection.')
                print(f'Rate {r} has the original source:\n{r.original_source}')
                print(f'Rate {r} is in chapter {r.chapter}')
        return len(set(names)) == len(self.rates)

    def _make_distinguishable(self, precedence):
        """If multiple rates have the same name, eliminate the extraneous ones according to their
        labels' positions in the precedence list. Only do this if all of the labels have
        rankings in the list."""

        nameset = {r.fname for r in self.rates}
        precedence = {lab: i for i, lab in enumerate(precedence)}
        def sorting_key(i):
            return precedence[self.rates[i].label]

        for n in nameset:

            # Count instances of name, and cycle if there is only one
            ind = [i for i, r in enumerate(self.rates) if r.fname == n]
            k = len(ind)
            if k <= 1:
                continue

            # If there were multiple instances, use the precedence settings to delete extraneous
            # rates
            labels = [self.rates[i].label for i in ind]

            if all(lab in precedence for lab in labels):

                sorted_ind = sorted(ind, key=sorting_key)
                r = self.rates[sorted_ind[0]]
                for i in sorted(sorted_ind[1:], reverse=True):
                    del self.rates[i]
                print(f'Found rate {r} named {n} with {k} entries in the RateCollection.')
                print(f'Kept only entry with label {r.label} out of {labels}.')

    def _write_network(self, *args, **kwargs):
        """A stub for function to output the network -- this is implementation
        dependent."""
        print('To create network integration source code, use a class that implements a specific network type.')

    def plot(self, outfile=None, rho=None, T=None, comp=None,
             size=(800, 600), dpi=100, title=None,
             ydot_cutoff_value=None,
             node_size=1000, node_font_size=13, node_color="#A0CBE2", node_shape="o",
             N_range=None, Z_range=None, rotated=False,
             always_show_p=False, always_show_alpha=False, hide_xalpha=False, filter_function=None):
        """Make a plot of the network structure showing the links between
        nuclei.  If a full set of thermodymamic conditions are
        provided (rho, T, comp), then the links are colored by rate
        strength.


        parameters
        ----------

        outfile: output name of the plot -- extension determines the type

        rho: density to evaluate rates with

        T: temperature to evaluate rates with

        comp: composition to evaluate rates with

        size: tuple giving width x height of the plot in inches

        dpi: pixels per inch used by matplotlib in rendering bitmap

        title: title to display on the plot

        ydot_cutoff_value: rate threshold below which we do not show a
        line corresponding to a rate

        node_size: size of a node

        node_font_size: size of the font used to write the isotope in the node

        node_color: color to make the nodes

        node_shape: shape of the node (using matplotlib marker names)

        N_range: range of neutron number to zoom in on

        Z_range: range of proton number to zoom in on

        rotate: if True, we plot A - 2Z vs. Z instead of the default Z vs. N

        always_show_p: include p as a node on the plot even if we
        don't have p+p reactions

        always_show_alpha: include He4 as a node on the plot even if we don't have 3-alpha

        hide_xalpha=False: dont connect the links to alpha for heavy
        nuclei reactions of the form A(alpha,X)B or A(X,alpha)B, except if alpha
        is the heaviest product.

        filter_function: name of a custom function that takes the list
        of nuclei and returns a new list with the nuclei to be shown
        as nodes.

        """

        G = nx.MultiDiGraph()
        G.position = {}
        G.labels = {}

        fig, ax = plt.subplots()
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes('right', size='15%', pad=0.05)

        #ax.plot([0, 0], [8, 8], 'b-')

        # in general, we do not show p, n, alpha,
        # unless we have p + p, 3-a, etc.
        hidden_nuclei = ["n"]
        if not always_show_p:
            hidden_nuclei.append("p")
        if not always_show_alpha:
            hidden_nuclei.append("he4")

        # nodes -- the node nuclei will be all of the heavies
        # add all the nuclei into G.node
        node_nuclei = []
        for n in self.unique_nuclei:
            if n.raw not in hidden_nuclei:
                node_nuclei.append(n)
            else:
                for r in self.rates:
                    if r.reactants.count(n) > 1:
                        node_nuclei.append(n)
                        break

        if filter_function is not None:
            node_nuclei = list(filter(filter_function, node_nuclei))

        for n in node_nuclei:
            G.add_node(n)
            if rotated:
                G.position[n] = (n.Z, n.A - 2*n.Z)
            else:
                G.position[n] = (n.N, n.Z)
            G.labels[n] = fr"${n.pretty}$"

        # get the rates for each reaction
        if rho is not None and T is not None and comp is not None:
            ydots = self.evaluate_rates(rho, T, comp)
        else:
            ydots = None

        # Do not show rates on the graph if their corresponding ydot is less than ydot_cutoff_value
        invisible_rates = set()
        if ydot_cutoff_value is not None:
            for r in self.rates:
                if ydots[r] < ydot_cutoff_value:
                    invisible_rates.add(r)

        # edges
        for n in node_nuclei:
            for r in self.nuclei_consumed[n]:
                for p in r.products:

                    if p in node_nuclei:

                        if hide_xalpha:

                            # first check is alpha is the heaviest nucleus on the RHS
                            rhs_heavy = sorted(r.products)[-1]
                            if not (rhs_heavy.Z == 2 and rhs_heavy.A == 4):

                                # for rates that are A (x, alpha) B, where A and B are heavy nuclei,
                                # don't show the connection of the nucleus to alpha, only show it to B
                                if p.Z == 2 and p.A == 4:
                                    continue

                                # likewise, hide A (alpha, x) B, unless A itself is an alpha
                                c = r.reactants
                                n_alpha = 0
                                for nuc in c:
                                    if nuc.Z == 2 and nuc.A == 4:
                                        n_alpha += 1
                                # if there is only 1 alpha and we are working on the alpha node,
                                # then skip
                                if n_alpha == 1 and n.Z == 2 and n.A == 4:
                                    continue

                        # networkx doesn't seem to keep the edges in
                        # any particular order, so we associate data
                        # to the edges here directly, in this case,
                        # the reaction rate, which will be used to
                        # color it
                        if ydots is None:
                            G.add_edges_from([(n, p)], weight=0.5)
                        else:
                            if r in invisible_rates:
                                continue
                            try:
                                rate_weight = math.log10(ydots[r])
                            except ValueError:
                                # if ydots[r] is zero, then set the weight
                                # to roughly the minimum exponent possible
                                # for python floats
                                rate_weight = -308
                            except:
                                raise
                            G.add_edges_from([(n, p)], weight=rate_weight)

        # It seems that networkx broke backwards compatability, and 'zorder' is no longer a valid
        # keyword argument. The 'linewidth' argument has also changed to 'linewidths'.

        nx.draw_networkx_nodes(G, G.position,      # plot the element at the correct position
                               node_color=node_color, alpha=1.0,
                               node_shape=node_shape, node_size=node_size, linewidths=2.0, ax=ax)

        nx.draw_networkx_labels(G, G.position, G.labels,   # label the name of element at the correct position
                                font_size=node_font_size, font_color="w", ax=ax)

        # get the edges and weights coupled in the same order
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

        edge_color=weights
        ww = np.array(weights)
        min_weight = ww.min()
        max_weight = ww.max()
        dw = (max_weight - min_weight)/4
        widths = np.ones_like(ww)
        widths[ww > min_weight + dw] = 1.5
        widths[ww > min_weight + 2*dw] = 2.5
        widths[ww > min_weight + 3*dw] = 4

        edges_lc = nx.draw_networkx_edges(G, G.position, width=list(widths),    # plot the arrow of reaction
                                          edgelist=edges, edge_color=edge_color,
                                          node_size=node_size,
                                          edge_cmap=plt.cm.viridis, ax=ax)

        # for networkx <= 2.0 draw_networkx_edges returns a
        # LineCollection matplotlib type which we can use for the
        # colorbar directly.  For networkx >= 2.1, it is a collection
        # of FancyArrowPatch-s, which we need to run through a
        # PatchCollection.  See:
        # https://stackoverflow.com/questions/18658047/adding-a-matplotlib-colorbar-from-a-patchcollection

        if ydots is not None:
            pc = mpl.collections.PatchCollection(edges_lc, cmap=plt.cm.viridis)
            pc.set_array(weights)
            if not rotated:
                plt.colorbar(pc, ax=ax, label="log10(rate)")
            else:
                plt.colorbar(pc, ax=ax, label="log10(rate)", orientation="horizontal", fraction=0.05)

        Ns = [n.N for n in node_nuclei]
        Zs = [n.Z for n in node_nuclei]

        if not rotated:
            ax.set_xlim(min(Ns)-1, max(Ns)+1)
        else:
            ax.set_xlim(min(Zs)-1, max(Zs)+1)

        #plt.ylim(min(Zs)-1, max(Zs)+1)

        if not rotated:
            plt.xlabel(r"$N$", fontsize="large")
            plt.ylabel(r"$Z$", fontsize="large")
        else:
            plt.xlabel(r"$Z$", fontsize="large")
            plt.ylabel(r"$A - 2Z$", fontsize="large")

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if Z_range is not None and N_range is not None:
            if not rotated:
                ax.set_xlim(N_range[0], N_range[1])
                ax.set_ylim(Z_range[0], Z_range[1])
            else:
                ax.set_xlim(Z_range[0], Z_range[1])

        if not rotated:
            ax.set_aspect("equal", "datalim")

        fig.set_size_inches(size[0]/dpi, size[1]/dpi)

        if title is not None:
            fig.suptitle(title)

        if outfile is None:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(outfile, dpi=dpi)

    @staticmethod
    def _safelog(arr, small):

        arr = np.copy(arr)
        if np.any(arr < 0.0):
            raise ValueError("Negative values not allowed for logscale - try symlog instead.")
        zeros = arr == 0.0
        arr[zeros] = min(small, arr[~zeros].min() / 10)
        return np.log10(arr)

    @staticmethod
    def _symlog(arr, linthresh=1.0):

        assert linthresh >= 1.0
        neg = arr < 0.0
        arr = np.abs(arr)
        needslog = arr > linthresh

        arr[needslog] = np.log10(arr[needslog]) + linthresh
        arr[neg] *= -1
        return arr

    @staticmethod
    def _scale(arr, minval=None, maxval=None):

        if minval is None:
            minval = arr.min()
        if maxval is None:
            maxval = arr.max()
        if minval != maxval:
            scaled = (arr - minval) / (maxval - minval)
        else:
            scaled = np.zeros_like(arr)
        scaled[scaled < 0.0] = 0.0
        scaled[scaled > 1.0] = 1.0
        return scaled

    def gridplot(self, comp=None, color_field="X", rho=None, T=None, **kwargs):
        """
        Plot nuclides as cells on a grid of Z vs. N, colored by *color_field*. If called
        without a composition, the function will just plot the grid with no color field.

        :param comp: Composition of the environment.
        :param color_field: Field to color by. Must be one of 'X' (mass fraction),
            'Y' (molar abundance), 'Xdot' (time derivative of X), 'Ydot' (time
            derivative of Y), or 'activity' (sum of contributions to Ydot of
            all rates, ignoring sign).
        :param rho: Density to evaluate rates at. Needed for fields involving time
            derivatives.
        :param T: Temperature to evaluate rates at. Needed for fields involving time
            derivatives.

        :Keyword Arguments:

            - *scale* -- One of 'linear', 'log', and 'symlog'. Linear by default.
            - *small* -- If using logarithmic scaling, zeros will be replaced with
              this value. 1e-30 by default.
            - *linthresh* -- Linearity threshold for symlog scaling.
            - *filter_function* -- A callable to filter Nucleus objects with. Should
              return *True* if the nuclide should be plotted.
            - *outfile* -- Output file to save the plot to. The plot will be shown if
              not specified.
            - *dpi* -- DPI to save the image file at.
            - *cmap* -- Name of the matplotlib colormap to use. Default is 'magma'.
            - *edgecolor* -- Color of grid cell edges.
            - *area* -- Area of the figure without the colorbar, in square inches. 64
              by default.
            - *no_axes* -- Set to *True* to omit axis spines.
            - *no_ticks* -- Set to *True* to omit tickmarks.
            - *no_cbar* -- Set to *True* to omit colorbar.
            - *cbar_label* -- Colorbar label.
            - *cbar_bounds* -- Explicit colorbar bounds.
            - *cbar_format* -- Format string or Formatter object for the colorbar ticks.
        """

        # Process kwargs
        outfile = kwargs.pop("outfile", None)
        scale = kwargs.pop("scale", "linear")
        cmap = kwargs.pop("cmap", "viridis")
        edgecolor = kwargs.pop("edgecolor", "grey")
        small = kwargs.pop("small", 1e-30)
        area = kwargs.pop("area", 64)
        no_axes = kwargs.pop("no_axes", False)
        no_ticks = kwargs.pop("no_ticks", False)
        no_cbar = kwargs.pop("no_cbar", False)
        cbar_label = kwargs.pop("cbar_label", None)
        cbar_format = kwargs.pop("cbar_format", None)
        cbar_bounds = kwargs.pop("cbar_bounds", None)
        filter_function = kwargs.pop("filter_function", None)
        dpi = kwargs.pop("dpi", 100)
        linthresh = kwargs.pop("linthresh", 1.0)

        if kwargs:
            warnings.warn(f"Unrecognized keyword arguments: {kwargs.keys()}")

        # Get figure, colormap
        fig, ax = plt.subplots()
        cmap = mpl.cm.get_cmap(cmap)

        # Get nuclei and all 3 numbers
        nuclei = self.unique_nuclei
        if filter_function is not None:
            nuclei = list(filter(filter_function, nuclei))
        Ns = np.array([n.N for n in nuclei])
        Zs = np.array([n.Z for n in nuclei])
        As = Ns + Zs

        # Compute weights
        color_field = color_field.lower()
        if color_field not in {"x", "y", "ydot", "xdot", "activity"}:
            raise ValueError(f"Invalid color field: '{color_field}'")

        if comp is None:

            values = np.zeros(len(nuclei))

        elif color_field == "x":

            values = np.array([comp.X[nuc] for nuc in nuclei])

        elif color_field == "y":

            ys = comp.get_molar()
            values = np.array([ys[nuc] for nuc in nuclei])

        elif color_field in {"ydot", "xdot"}:

            if rho is None or T is None:
                raise ValueError("Need both rho and T to evaluate rates!")
            ydots = self.evaluate_ydots(rho, T, comp)
            values = np.array([ydots[nuc] for nuc in nuclei])
            if color_field == "xdot":
                values *= As

        elif color_field == "activity":

            if rho is None or T is None:
                raise ValueError("Need both rho and T to evaluate rates!")
            act = self.evaluate_activity(rho, T, comp)
            values = np.array([act[nuc] for nuc in nuclei])

        if scale == "log":
            values = self._safelog(values, small)
        elif scale == "symlog":
            values = self._symlog(values, linthresh)

        if cbar_bounds is None:
            cbar_bounds = values.min(), values.max()
        weights = self._scale(values, *cbar_bounds)

        # Plot a square for each nucleus
        for nuc, weight in zip(nuclei, weights):

            square = plt.Rectangle((nuc.N - 0.5, nuc.Z - 0.5), width=1, height=1,
                    facecolor=cmap(weight), edgecolor=edgecolor)
            ax.add_patch(square)

        # Set limits
        maxN, minN = max(Ns), min(Ns)
        maxZ, minZ = max(Zs), min(Zs)

        plt.xlim(minN - 0.5, maxN + 0.6)
        plt.ylim(minZ - 0.5, maxZ + 0.6)

        # Set plot appearance
        rat = (maxN - minN) / (maxZ - minZ)
        width = np.sqrt(area * rat)
        height = area / width
        fig.set_size_inches(width, height)

        plt.xlabel(r"N $\rightarrow$")
        plt.ylabel(r"Z $\rightarrow$")

        if no_axes or no_ticks:

            plt.tick_params \
            (
                axis = 'both',
                which = 'both',
                bottom = False,
                left = False,
                labelbottom = False,
                labelleft = False
            )

        else:

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if no_axes:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        # Colorbar stuff
        if not no_cbar and comp is not None:

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3.5%', pad=0.1)
            cbar_norm = mpl.colors.Normalize(*cbar_bounds)
            smap = mpl.cm.ScalarMappable(norm=cbar_norm, cmap=cmap)

            if not cbar_label:

                capfield = color_field.capitalize()
                if scale == "log":
                    cbar_label = f"log[{capfield}]"
                elif scale == "symlog":
                    cbar_label = f"symlog[{capfield}]"
                else:
                    cbar_label = capfield

            fig.colorbar(smap, cax=cax, orientation="vertical",
                    label=cbar_label, format=cbar_format)

        # Show or save
        if outfile is None:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(outfile, dpi=dpi)

    def __repr__(self):
        string = ""
        for r in self.rates:
            string += f"{r.string}\n"
        return string


class Explorer:
    """ interactively explore a rate collection """
    def __init__(self, rc, comp, size=(800, 600),
                 ydot_cutoff_value=None,
                 always_show_p=False, always_show_alpha=False):
        """ take a RateCollection and a composition """
        self.rc = rc
        self.comp = comp
        self.size = size
        self.ydot_cutoff_value = ydot_cutoff_value
        self.always_show_p = always_show_p
        self.always_show_alpha = always_show_alpha

    def _make_plot(self, logrho, logT):
        self.rc.plot(rho=10.0**logrho, T=10.0**logT,
                     comp=self.comp, size=self.size,
                     ydot_cutoff_value=self.ydot_cutoff_value,
                     always_show_p=self.always_show_p,
                     always_show_alpha=self.always_show_alpha)

    def explore(self, logrho=(2, 6, 0.1), logT=(7, 9, 0.1)):
        """Perform interactive exploration of the network structure."""
        interact(self._make_plot, logrho=logrho, logT=logT)



class ChemRateCollection:
    """ a collection of rates that together define a network """

    pynucastro_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    def __init__(self, rates=None, tdot_switch=0):
        """rate_files are the files that together define the network.  This
        can be any iterable or single string.

        If rates is supplied, initialize a RateCollection using the
        Rate objects in the list 'rates'.

        tdot_switch = 0 --> tdot are dT /dt
        tdot_switch = 1 --> tdot are dEint /dt

        """

        if isinstance(rates, ChemRate):
            rates = [rates]

        try:
            for r in rates:
                assert isinstance(r, ChemRate)
        except:
            print('Expected ChemRate object or list of ChemRate objects passed as the rates argument.')
            raise

        self.rates = rates
        self.tdot_switch = tdot_switch

    def get_allspecies(self):
        # get the unique species
        u = []
        for r in self.rates:
            u.append(r.reactants)
            u.append(r.products)

        #make a flattened list of u
        cc = [item for sublist in u for item in sublist]
        #to get unique species out, make it a set
        #sort it by the order of specie mass to ensure RHS and LHS are consistently ordered during network integration
        return sorted(list(set(cc)))

    def evaluate_rates(self, T, composition):
        rvals = []
        #y_e = ChemComposition.eval_ye()
        for r in self.rates:
            val = r.eval(T, composition)
            rvals.append(val)

        return rvals


    
    def evaluate_gamma(self, composition):

        #gass = 0
        #nmols = self.get_n(composition)
        #for specie in self.get_allspecies():
        #    gass += composition[specie]/(specie.gamma - 1.0)

        #gamma_index = 1.0 + nmols/gass

        gamma_index = (5.0*(composition[ChemSpecie('elec')] + composition[ChemSpecie('h')] + composition[ChemSpecie('he')]) + \
                       7.0*(composition[ChemSpecie('h2')])) / (3.0*(composition[ChemSpecie('elec')] + composition[ChemSpecie('h')] + \
                                                                    composition[ChemSpecie('he')]) + 5.0*(composition[ChemSpecie('h2')]))

        return gamma_index


    def evaluate_ydots(self, T, composition):
        """evaluate net rate of change of molar abundance for each nucleus
        for a specific density, temperature, and composition"""

        ydots = dict()

        for specie in self.get_allspecies():
            ydots[specie] = 0
            for r in self.rates:
                if specie in r.reactants:
                    
                    #ydots[specie] += -r.eval(T, composition) * Counter(r.reactants)[specie] * \
                    #                  (composition[specie]**Counter(r.reactants)[specie]) * \
                    #                  functools.reduce(mul, [composition[q] for q in list(filter((specie).__ne__, r.reactants)) ], 1)

                    ydots[specie] += -r.eval(T, composition) * Counter(r.reactants)[specie] * \
                                      functools.reduce(mul, [composition[q] for q in r.reactants])

                    #if specie == ChemSpecie('h'):
                    #    print(-Counter(r.reactants)[specie], r)

                if specie in r.products:
                    ydots[specie] += r.eval(T, composition) * Counter(r.products)[specie] * \
                                     functools.reduce(mul, [composition[q] for q in r.reactants])
                    
                    #if specie == ChemSpecie('h'):
                    #    print(Counter(r.products)[specie], r)

        return ydots

    def evaluate_cooling(self, T, composition, redshift):
        #NOTE - CIE cooling not used as per krome's test
        sumcool = self.evaluate_cooling_cont(T, composition) + self.evaluate_cooling_compton(T, composition, redshift) + \
                  self.evaluate_cooling_chem(T, composition) + self.evaluate_cooling_atomic(T, composition) + \
                  self.evaluate_cooling_H2(T, composition)

        if ChemSpecie('hd') in composition:
            sumcool += self.evaluate_cooling_HD(T, composition, redshift)

        return sumcool

    def evaluate_heating(self, T, composition, redshift):
        sumheat = self.evaluate_heating_chem(T, composition) + self.evaluate_heating_compress(T, composition)

        return sumheat

    def evaluate_tdot(self, T, composition, redshift):
        if self.tdot_switch == 0:
            #find dT/dt
            tdot = (self.evaluate_heating(T, composition, redshift) - self.evaluate_cooling(T, composition, redshift)) * \
                   (self.evaluate_gamma(composition) - 1.0) / cons.boltzmann_erg / self.get_n(composition)
        elif self.tdot_switch == 1:
            #find dEint/dt
            tdot = (self.evaluate_heating(T, composition, redshift) - self.evaluate_cooling(T, composition, redshift)) / self.get_rho(composition)
        else:
            raise ValueError('Incorrect value for tdot_switch!')
        return tdot
    
    def get_Hnuclei(self, composition):

        nH = composition[ChemSpecie('hp')] + composition[ChemSpecie('h')] + composition[ChemSpecie('hm')] + \
             composition[ChemSpecie('h2')]*2.0 + composition[ChemSpecie('h2p')]*2.0

        if ChemSpecie('hd') in composition:
            nH += composition[ChemSpecie('hd')] + composition[ChemSpecie('hdp')]

        return nH

    def get_n(self, composition):

        n = composition[ChemSpecie('hp')] + composition[ChemSpecie('h')] + composition[ChemSpecie('hm')] + \
            composition[ChemSpecie('h2')] + composition[ChemSpecie('h2p')] + composition[ChemSpecie('hepp')] + \
            composition[ChemSpecie('he')] + composition[ChemSpecie('hep')] + composition[ChemSpecie('elec')]
            
        if ChemSpecie('hd') in composition:
            n += composition[ChemSpecie('hd')] + composition[ChemSpecie('hdp')]

        if ChemSpecie('d') in composition:
            n += composition[ChemSpecie('dm')] + composition[ChemSpecie('d')] + composition[ChemSpecie('dp')]

        return n

    def get_rho(self, composition):

        rho = composition[ChemSpecie('hp')]*ChemSpecie('hp').m + composition[ChemSpecie('h')]*ChemSpecie('h').m + \
              composition[ChemSpecie('hm')]*ChemSpecie('hm').m + composition[ChemSpecie('h2')]*ChemSpecie('h2').m + \
              composition[ChemSpecie('h2p')]*ChemSpecie('h2p').m + composition[ChemSpecie('he')]*ChemSpecie('he').m + \
              composition[ChemSpecie('hep')]*ChemSpecie('hep').m + composition[ChemSpecie('hepp')]*ChemSpecie('hepp').m + \
              composition[ChemSpecie('elec')]*ChemSpecie('elec').m             

        if ChemSpecie('hd') in composition:
            rho += composition[ChemSpecie('hd')]*ChemSpecie('hd').m + composition[ChemSpecie('hdp')]*ChemSpecie('hdp').m

        if ChemSpecie('d') in composition:
            rho += composition[ChemSpecie('d')]*ChemSpecie('d').m + composition[ChemSpecie('dm')]*ChemSpecie('dm').m + \
                   composition[ChemSpecie('dp')]*ChemSpecie('dp').m

        return rho

    def evaluate_heating_chem(self, T, composition):

        dd = self.get_Hnuclei(composition)
        small = 1e-99
        heatingChem = 0.

        ncrn  = 1.0e6*(T**(-0.5))
        ncrd1 = 1.6*np.exp(-(4.0e2/T)**2)
        ncrd2 = 1.4*np.exp(-1.2e4/(T+1.2e3))

        yH = composition[ChemSpecie('h')]/dd
        yH2 = composition[ChemSpecie('h2')]/dd
        ncr = ncrn/(ncrd1*yH + ncrd2*yH2)
        h2heatfac = 1.0/(1.0 + ncr/dd)

        HChem = 0.
        a1, b1, c1, d1 = 0., 0., 0., 0.

        for r in self.rates:
            if {ChemSpecie('hm'), ChemSpecie('h')} == set(r.reactants) and {ChemSpecie('h2'), ChemSpecie('elec')} == set(r.products):
                #reaction 10
                a1 = ChemRate(reactants=r.reactants, products=r.products).eval(T, composition) * \
                         (3.53*h2heatfac*functools.reduce(mul, [composition[q] for q in r.reactants]))

        for r in self.rates:
            if {ChemSpecie('h2p'), ChemSpecie('h')} == set(r.reactants) and {ChemSpecie('hp'), ChemSpecie('h2')} == set(r.products):
                #reaction 13
                b1 = ChemRate(reactants=r.reactants, products=r.products).eval(T, composition) * \
                         (1.83*h2heatfac*functools.reduce(mul, [composition[q] for q in r.reactants]))

        for r in self.rates:
            if Counter(r.reactants)[ChemSpecie('h')] == 3 and {ChemSpecie('h2'), ChemSpecie('h')} == set(r.products):
                #reaction 25
                c1 = ChemRate(reactants=r.reactants, products=r.products).eval(T, composition) * \
                         (4.48*h2heatfac*functools.reduce(mul, [composition[q] for q in r.reactants]))

        for r in self.rates:
            if Counter(r.reactants)[ChemSpecie('h')] == 2 and Counter(r.reactants)[ChemSpecie('h2')] == 1 and Counter(r.products)[ChemSpecie('h2')] == 2:
                #reaction 26
                d1 = ChemRate(reactants=r.reactants, products=r.products).eval(T, composition) * \
                         (4.48*h2heatfac*functools.reduce(mul, [composition[q] for q in r.reactants]))

        HChem = a1 + b1 + c1 + d1
        return HChem*cons.eV_to_erg

    def get_free_fall_time(self, composition):
        rhogas = self.get_rho(composition)
        free_fall_time = np.sqrt(3.0*np.pi/32.0/cons.gravity/rhogas)
        return free_fall_time

    def evaluate_heating_compress(self, T, composition):

        free_fall_time = self.get_free_fall_time(composition)

        dd = self.get_n(composition)

        Hcompress = dd * cons.boltzmann_erg * T / free_fall_time #erg/s/cm3
        return Hcompress


    def evaluate_cooling_chem(self, T, composition):

        CChem = 0.
        a1, b1, c1 = 0., 0., 0.
        for r in self.rates:
            if {ChemSpecie('h2'), ChemSpecie('elec')} == set(r.reactants) and Counter(r.products)[ChemSpecie('h')] == 2 and Counter(r.products)[ChemSpecie('elec')] == 1:
                #reaction 16
                a1 = ChemRate(reactants=r.reactants, products=r.products).eval(T, composition) * \
                         (4.48*functools.reduce(mul, [composition[q] for q in r.reactants]))

        for r in self.rates:
            if {ChemSpecie('h2'), ChemSpecie('h')} == set(r.reactants) and Counter(r.products)[ChemSpecie('h')] == 3:
                #reaction 17
                b1 = ChemRate(reactants=r.reactants, products=r.products).eval(T, composition) * \
                         (4.48*functools.reduce(mul, [composition[q] for q in r.reactants]))

        for r in self.rates:
            if Counter(r.reactants)[ChemSpecie('h2')] == 2 and Counter(r.products)[ChemSpecie('h')] == 2 and Counter(r.products)[ChemSpecie('h2')] == 1:
                #reaction 27
                c1 = ChemRate(reactants=r.reactants, products=r.products).eval(T, composition) * \
                         (4.48*functools.reduce(mul, [composition[q] for q in r.reactants]))

        CChem = a1 + b1 + c1

        return CChem*cons.eV_to_erg       
    
    def evaluate_cooling_atomic(self, T, composition):
        temp = max(T,10) #K
        T5 = temp/1e5 
        
        Catomic = 0.

        #COLLISIONAL IONIZATION: H, He, He+, He(2S)
        Catomic = Catomic+ 1.27e-21*np.sqrt(temp)/(1.0+np.sqrt(T5))*np.exp(-1.578091e5/temp)*composition[ChemSpecie('elec')] * \
                  composition[ChemSpecie('h')]

        Catomic = Catomic + 9.38e-22*np.sqrt(temp)/(1.0+np.sqrt(T5))*np.exp(-2.853354e5/temp)*composition[ChemSpecie('elec')] * \
                  composition[ChemSpecie('he')]

        Catomic = Catomic + 4.95e-22*np.sqrt(temp)/(1.0+np.sqrt(T5))*np.exp(-6.31515e5/temp)*composition[ChemSpecie('elec')] * \
                  composition[ChemSpecie('hep')]
        Catomic = Catomic + 5.01e-27*temp**(-0.1687)/(1.0+np.sqrt(T5))*np.exp(-5.5338e4/temp)*composition[ChemSpecie('elec')]**2 * \
                  composition[ChemSpecie('hep')]


        #RECOMBINATION: H+, He+,He2+
        Catomic = Catomic + 8.7e-27*np.sqrt(temp)*(temp/1.e3)**(-0.2)/(1.0+(temp/1.e6)**0.7)*composition[ChemSpecie('elec')] * \
                  composition[ChemSpecie('hp')]

        Catomic = Catomic + 1.55e-26*temp**(0.3647)*composition[ChemSpecie('elec')]*composition[ChemSpecie('hep')]

        Catomic = Catomic + 3.48e-26*np.sqrt(temp)*(temp/1.e3)**(-0.2)/(1.0+(temp/1.e6)**0.7)*composition[ChemSpecie('elec')] * \
                  composition[ChemSpecie('hepp')]


        #!DIELECTRONIC RECOMBINATION: He
        Catomic = Catomic + 1.24e-13*temp**(-1.5)*np.exp(-4.7e5/temp)*(1.0+0.30*np.exp(-9.4e4/temp))*composition[ChemSpecie('elec')] * \
                  composition[ChemSpecie('hep')]


        #COLLISIONAL EXCITATION:
        #H(all n), He(n=2,3,4 triplets), He+(n=2)
        Catomic = Catomic + 7.5e-19/(1.0+np.sqrt(T5))*np.exp(-1.18348e5/temp)*composition[ChemSpecie('elec')] * \
                  composition[ChemSpecie('h')]

        Catomic = Catomic + 9.1e-27*temp**(-.1687)/(1.0+np.sqrt(T5))*np.exp(-1.3179e4/temp)*composition[ChemSpecie('elec')]**2 * \
                  composition[ChemSpecie('hep')]
        Catomic = Catomic + 5.54e-17*temp**(-.397)/(1.0+np.sqrt(T5))*np.exp(-4.73638e5/temp)*composition[ChemSpecie('elec')] * \
                  composition[ChemSpecie('hep')]

        return Catomic

    def evaluate_cooling_compton(self, T, composition, redshift):
        Ccompton = 5.65e-36 * (1.0 + redshift)**4 * (T - 2.73 * (1.0 + redshift)) * composition[ChemSpecie('elec')] #erg/s/cm3
        return Ccompton

    def kpla(self, composition):
        rhogas = self.get_rho(composition)

        kpla = 0.0
        
        #opacity is zero under 1e-12 g/cm3
        if rhogas < 1e-12:
            return kpla

        a0 = 1.000042e0
        a1 = 2.14989e0

        #log density cannot exceed 0.5 g/cm3
        y = np.log10(min(rhogas,0.50))

        kpla = 1e1**(a0*y + a1) #fit density only
        return kpla

    def get_jeans_length(self, T, composition):

        rhogas = max(self.get_rho(composition), 1e-40)
        mu = rhogas / max(self.get_n(composition), 1e-40) * cons.ip_mass
        get_jeans_length = np.sqrt(np.pi * cons.boltzmann_erg * T/rhogas / cons.p_mass / cons.gravity / mu)
        return get_jeans_length

    def evaluate_cooling_cont(self, T, composition):

        rhogas = self.get_rho(composition) #g/cm3
        kgas = self.kpla(composition) #planck opacity cm2/g (Omukai+2000)
        lj = self.get_jeans_length(T, composition) #cm
        tau = lj * kgas * rhogas + 1e-40 #opacity
        beta = min(1.0, tau**(-2)) #beta escape (always <1.)
        Ccont = 4.0 * cons.stefboltz_erg * (T**4) * kgas * rhogas * beta #erg/s/cm3
        return Ccont

    def evaluate_cooling_CIE(self, T, composition, redshift):
        CCIE = 0.0
        Tcmb = 2.73*(1+redshift)
        #set cooling to zero if n_H2 is smaller than 1e-12 1/cm3
        #to avoid division by zero in opacity term due to tauCIE=0
        if composition[ChemSpecie('h2')] < 1e-12:
            return 0.

        if T < Tcmb:
            return 0.

        x = np.log10(T)
        x2 = x*x
        x3 = x2*x
        x4 = x3*x
        x5 = x4*x

        cool = 0.0
        #outside boundaries below cooling is zero
        logcool = -1e99

        #evaluates fitting functions
        if x > 2.0 and x < 2.95:
            a0 = -30.3314216559651
            a1 = 19.0004016698518
            a2 = -17.1507937874082
            a3 = 9.49499574218739
            a4 = -2.54768404538229
            a5 = 0.265382965410969
            logcool = a0 + a1*x + a2*x2 + a3*x3 +a4*x4 +a5*x5
        elif x >= 2.95 and x < 5:
            b0 = -180.992524120965
            b1 = 168.471004362887
            b2 = -67.499549702687
            b3 = 13.5075841245848
            b4 = -1.31983368963974
            b5 = 0.0500087685129987
            logcool = b0 + b1*x + b2*x2 + b3*x3 +b4*x4 +b5*x5
        elif x >= 5:
            logcool = 3.0 * x - 21.2968837223113 #cubic extrapolation

        #opacity according to RA04
        tauCIE = (composition[ChemSpecie('h2')] * 1.4285714e-16)**2.8 #note: 1/7d15 = 1.4285714d-16
        cool = cons.p_mass * 1e1**logcool #erg*cm3/s

        CCIE = cool * min(1.0, (1.0-np.exp(-tauCIE))/tauCIE) * composition[ChemSpecie('h2')] * self.get_n(composition) #erg/cm3/s

        return CCIE

    def evaluate_cooling_HD(self, T, composition, redshift):
        Tcmb = 2.73*(1+redshift)
        CHD = 0.0 #erg/cm3/s

        #this function does not have limits on density
        #and temperature, even if the original paper do.
        #However, we extrapolate the limits.

        #exit on low temperature
        if(T < Tcmb):
            return 0.0

        #extrapolate higher temperature limit
        Tgas = min(T,1e4)

        #calculate density
        dd = composition[ChemSpecie('h')] #self.get_n(composition)
        #exit if density is out of Lipovka bounds (uncomment if needed)
        #if(dd<1d0 .or. dd>1d8) return
        #extrapolate density limits
        dd = min(max(dd,1e-2),1e10)

        #POLYNOMIAL COEFFICIENT: TABLE 1 LIPOVKA
        lipovka = np.zeros((5,5))
        lipovka[0,:] = np.array([-42.56788, 0.92433, 0.54962, -0.07676, 0.00275])
        lipovka[1,:] = np.array([21.93385, 0.77952, -1.06447, 0.11864, -0.00366])
        lipovka[2,:] = np.array([-10.19097, -0.54263, 0.62343, -0.07366, 0.002514])
        lipovka[3,:] = np.array([2.19906, 0.11711, -0.13768, 0.01759, -0.00066631])
        lipovka[4,:] = np.array([-0.17334, -0.00835, 0.0106, -0.001482, 0.00006192])

        logTgas = np.log10(Tgas)
        lognH   = np.log10(dd)

        #loop to compute coefficients
        logW = 0.0
        for j in range(0,5):
            lHj = lognH**j
            for i in range(0,5):
                logW += lipovka[i,j]*logTgas**i*lHj #erg/s

        W = 10.0**(logW)
        CHD = W * composition[ChemSpecie('hd')] #erg/cm3/s
        return CHD


    def sigmoid(self, x, x0, s):
        sigmoid = 1e1/(1e1+np.exp(-s*(x-x0)))
        return sigmoid

    def wCool(self, logTgas, logTmin, logTmax):
        x = (logTgas-logTmin)/(logTmax-logTmin)
        wCool = 1e1**(2e2*(self.sigmoid(x,-2e-1,5e1)*self.sigmoid(-x,-1.2,5e1)-1.0))

        if wCool < 1e-199:
            wCool = 0.0
        if wCool > 1:
            raise ValueError("wCool > 1 in H2 cooling!")

        return wCool

    def evaluate_cooling_H2(self, T, composition):

        temp = T
        CH2 = 0.0
        #if(temp<2d0) return

        t3 = temp * 1e-3
        logt3 = np.log10(t3)
        logt = np.log10(temp)
        cool = 0.0

        logt32 = logt3 * logt3
        logt33 = logt32 * logt3
        logt34 = logt33 * logt3
        logt35 = logt34 * logt3
        logt36 = logt35 * logt3
        logt37 = logt36 * logt3
        logt38 = logt37 * logt3

        w14 = self.wCool(logt, 1.0, 4.0)
        w24 = self.wCool(logt, 2.0, 4.0)

        #//H2-H
        if temp <= 1e2:
            fH2H = 1.e1**(-16.818342 + 3.7383713e1*logt3 + 5.8145166e1*logt32 + \
                           4.8656103e1*logt33 + 2.0159831e1*logt34 + 3.8479610*logt35) * composition[ChemSpecie('h')]
        elif temp > 1e2 and temp <= 1e3:
            fH2H = 1.e1**(-2.4311209e1 + 3.5692468*logt3 -1.1332860e1*logt32 - 2.7850082e1*logt33 - \
                           2.1328264e1*logt34 - 4.2519023*logt35) * composition[ChemSpecie('h')]
        elif temp > 1e3 and temp <= 6e3:
            fH2H = 1e1**(-2.4311209e1 + 4.6450521*logt3 - 3.7209846*logt32 + 5.9369081*logt33 - \
                          5.5108049*logt34 + 1.5538288*logt35) * composition[ChemSpecie('h')]
        else:
            fH2H = 1.862314467912518e-022*self.wCool(logt,1.0,np.log10(6e3)) * composition[ChemSpecie('h')]

        cool += fH2H

        #//H2-Hp
        if temp > 1e1 and temp <= 1e4:
            fH2Hp = 1e1**(-2.2089523e1 +1.5714711*logt3 + 0.015391166*logt32 - 0.23619985*logt33 - \
                           0.51002221*logt34 + 0.32168730*logt35) * composition[ChemSpecie('hp')]
        else:
            fH2Hp = 1.182509139382060e-021 * composition[ChemSpecie('hp')] * w14

        cool += fH2Hp

        #//H2-H2
        fH2H2 = w24 * 1e1**(-2.3962112e1 + 2.09433740*logt3 - 0.77151436*logt32 + \
                             0.43693353*logt33 - 0.14913216*logt34 - 0.033638326*logt35) * composition[ChemSpecie('h2')]

        cool += fH2H2

        #//H2-e
        fH2e = 0.0
        if temp <= 5e2:
            fH2e = 1e1**(min(-2.1928796e1 + 1.6815730e1*logt3 + 9.6743155e1*logt32 + 3.4319180e2*logt33 + \
                              7.3471651e2*logt34 + 9.8367576e2*logt35 + 8.0181247e2*logt36 + \
                              3.6414446e2*logt37 + 7.0609154e1*logt38,3e1)) * composition[ChemSpecie('elec')]
        elif temp > 5e2:
            fH2e = 1e1**(-2.2921189e1 + 1.6802758*logt3 + 0.93310622*logt32 + 4.0406627*logt33 - \
                          4.7274036*logt34 - 8.8077017*logt35 + 8.9167183*logt36 + 6.4380698*logt37 - \
                          6.3701156*logt38) * composition[ChemSpecie('elec')]

        cool += fH2e*w24

        #//H2-He
        if temp > 1e1 and temp <= 1e4:
            fH2He = 1e1**(-2.3689237e1 +2.1892372*logt3 - 0.81520438*logt32 + 0.29036281*logt33 - \
                           0.16596184*logt34 + 0.19191375*logt35) * composition[ChemSpecie('he')]
        else:
            fH2He = 1.002560385050777e-022 * composition[ChemSpecie('he')] * w14

        cool += fH2He

        #check error
        if cool > 1e30:
            raise ValueError(" ERROR: cooling >1.d30 erg/s/cm3")

        #this to avoid negative, overflow and useless calculations below
        if cool <= 0.0:
            CH2 = 0.0
            return CH2

        #high density limit from HM79, GP98 below Tgas = 2d3
        #UPDATED USING GLOVER 2015 for high temperature corrections, MNRAS
        #IN THE HIGH DENSITY REGIME LAMBDA_H2 = LAMBDA_H2(LTE) = HDL
        #the following mix of functions ensures the right behaviour
        # at low (T<10 K) and high temperatures (T>2000 K) by
        # using both the original Hollenbach and the new Glover data
        # merged in a smooth way.
        if temp < 2e3:
            HDLR = ((9.5e-22*t3**3.76)/(1.+0.12*t3**2.1)*np.exp(-(0.13/t3)**3) + 3.e-24*np.exp(-0.51/t3)) #erg/s
            HDLV = (6.7e-19*np.exp(-5.86/t3) + 1.6e-18*np.exp(-11.7/t3)) #erg/s
            HDL  = HDLR + HDLV #erg/s
        elif temp >= 2e3 and temp <= 1e4:
            HDL = 1e1**(-2.0584225e1 + 5.0194035*logt3 - 1.5738805*logt32 - 4.7155769*logt33 + \
                         2.4714161*logt34 + 5.4710750*logt35 - 3.9467356*logt36 - 2.2148338*logt37 + 1.8161874*logt38)
        else:
            dump14 = 1.0 / (1.0 + np.exp(min((temp-3e4)*2e-4,3e2)))
            HDL = 5.531333679406485E-019*dump14

        LDL = cool #erg/s
        if HDL == 0.0:
            CH2 = 0.0
        else:
            CH2 = composition[ChemSpecie('h2')]/(1.0/HDL+1.0/LDL) * \
                  min(1.0, max(1.25e-10 * self.get_n(composition), 1e-40)**(-0.45)) #erg/cm3/s


        return CH2

    def fft(self, y):
        allspecies = self.get_allspecies()
        comp = ChemComposition(specie=(allspecies)).set_specie_numberdens(nval=(y))


        free_fall_time = self.get_free_fall_time(comp)
        return free_fall_time


    def rhs(self, t, y):
        redshift = 30.0
        allspecies = self.get_allspecies()
        comp = ChemComposition(specie=(allspecies)).set_specie_numberdens(nval=(y[:-1]))


        ydots = self.evaluate_ydots(y[len(y)-1], comp)

        #don't need to sort ydots since ydots are already sorted
        sorted_ydots = ydots #{key: value for key, value in sorted(ydots.items())}
        tdot = self.evaluate_tdot(y[len(y)-1], comp, redshift)
        bb = list(sorted_ydots.values())
        bb.append(tdot)

        return bb



class SympyChemRateCollection:
    """ a collection of rates that together define a network """

    pynucastro_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    def __init__(self, rates=None, tdot_switch=0, ydots_lambdified=False, jacs_lambdified=False, withD=1, massfracs=0):
        """rate_files are the files that together define the network.  This
        can be any iterable or single string.

        If rates is supplied, initialize a RateCollection using the
        Rate objects in the list 'rates'.

        tdot_switch = 0 --> tdot are dT /dt
        tdot_switch = 1 --> tdot are dEint /dt

        """
        import sympy as sp

        if isinstance(rates, SympyChemRate):
            rates = [rates]

        try:
            for r in rates:
                assert isinstance(r, SympyChemRate)
        except:
            print('Expected ChemRate object or list of ChemRate objects passed as the rates argument.')
            raise

        self.rates = rates
        self.tdot_switch = tdot_switch
        self.withD = withD
        self.massfracs = massfracs

        if (self.massfracs != 0) & (self.massfracs != 1):
            raise ValueError('The code only works with mass fractions or number densities!')

        if (self.tdot_switch != 0) & (self.tdot_switch != 1):
            raise ValueError('The code only works with temperature or specific internal energy!')

        if (ydots_lambdified):
            print('Using ydots_lambdified')
            redshift = 30
            self.ydots_lambdified = self.get_lambdifys_ydots(redshift)

        if (jacs_lambdified):
            print('Using jacs_lambdified')
            redshift = 30
            if tdot_switch == 0:
                self.jacs_lambdified = self.get_lambdifys_jacobian_T(redshift)
            elif tdot_switch == 1:
                self.jacs_lambdified = self.get_lambdifys_jacobian_Eint(redshift)

    def get_allspecies_sym(self):
        # get the unique species
        # HARDCODED for now

        if self.withD == 1:
            splist = [ChemSpecie('e').sym_name, ChemSpecie('hp').sym_name, ChemSpecie('h').sym_name, ChemSpecie('hm').sym_name, \
                      ChemSpecie('dp').sym_name, ChemSpecie('d').sym_name, ChemSpecie('h2p').sym_name, ChemSpecie('dm').sym_name, \
                      ChemSpecie('h2').sym_name, ChemSpecie('hdp').sym_name, ChemSpecie('hd').sym_name, ChemSpecie('hepp').sym_name, \
                      ChemSpecie('hep').sym_name, ChemSpecie('he').sym_name]
        elif self.withD == 0:
            splist = [ChemSpecie('e').sym_name, ChemSpecie('hp').sym_name, ChemSpecie('h').sym_name, ChemSpecie('hm').sym_name, \
                      ChemSpecie('h2p').sym_name, \
                      ChemSpecie('h2').sym_name, ChemSpecie('hepp').sym_name, \
                      ChemSpecie('hep').sym_name, ChemSpecie('he').sym_name]

        else:
            raise ValueError('Incorrect value of withD!')

        return splist

    def get_allspecies(self):
        # get the unique species
        # HARDCODED for now
        if self.withD == 1:
            splist = (ChemSpecie('e'), ChemSpecie('hp'), ChemSpecie('h'), ChemSpecie('hm'), \
                      ChemSpecie('dp'), ChemSpecie('d'), ChemSpecie('h2p'), ChemSpecie('dm'), \
                      ChemSpecie('h2'), ChemSpecie('hdp'), ChemSpecie('hd'), ChemSpecie('hepp'), \
                      ChemSpecie('hep'), ChemSpecie('he'))
        elif self.withD == 0:
            splist = (ChemSpecie('e'), ChemSpecie('hp'), ChemSpecie('h'), ChemSpecie('hm'), \
                      ChemSpecie('h2p'), ChemSpecie('h2'), ChemSpecie('hepp'), \
                      ChemSpecie('hep'), ChemSpecie('he'))
        else:
            raise ValueError('Incorrect value of withD!')

        return splist

    def sympy_to_python(self, sympy_specie):
        #given a sympy specie, it returns the chemspecie
        #this is needed when using functools.reduce(mul) operation while evaluating rates below in ydots
        allspecies =self.get_allspecies()
        saved = ChemSpecie('dummy')
        for specie in allspecies:
            if specie.sym_name == sympy_specie:
                saved = specie
                break

        if saved == ChemSpecie('dummy'):
            raise ValueError('Converting sympy symbol of a specie back to ChemSpecie object failed!')

        return saved

    def evaluate_rates(self, T, composition, density=0):
        rvals = []
        #y_e = ChemComposition.eval_ye()
        for r in self.rates:
            val = r.eval(T, composition, density)
            rvals.append(val)

        return rvals
    
    def evaluate_gamma(self, composition, density=0):

        if self.massfracs == 1:
            gamma_index = (5.0*density*(composition[ChemSpecie('elec').sym_name]/ChemSpecie('elec').m + composition[ChemSpecie('h').sym_name]/ChemSpecie('h').m + composition[ChemSpecie('he').sym_name]/ChemSpecie('he').m) + \
                           7.0*density*(composition[ChemSpecie('h2').sym_name]/ChemSpecie('h2').m)) / (3.0*density*(composition[ChemSpecie('elec').sym_name]/ChemSpecie('elec').m + composition[ChemSpecie('h').sym_name]/ChemSpecie('h').m + \
                                                                        composition[ChemSpecie('he').sym_name]/ChemSpecie('he').m) + 5.0*density*(composition[ChemSpecie('h2').sym_name]/ChemSpecie('h2').m))

        elif self.massfracs == 0:
            gamma_index = (5.0*(composition[ChemSpecie('elec').sym_name] + composition[ChemSpecie('h').sym_name] + composition[ChemSpecie('he').sym_name]) + \
                           7.0*composition[ChemSpecie('h2').sym_name]) / (3.0*(composition[ChemSpecie('elec').sym_name] + composition[ChemSpecie('h').sym_name] + \
                                                                        composition[ChemSpecie('he').sym_name]) + 5.0*composition[ChemSpecie('h2').sym_name])

        return gamma_index


    def evaluate_ydots(self, T, composition, density=0):
        """evaluate net rate of change of mass fraction abundance for each nucleus
        for a specific density, temperature, and composition"""

        ydots = dict()

        for specie in self.get_allspecies_sym():
            ydots[specie] = 0
            for r in self.rates:
                if specie in r.reactants:
                    
                    if self.massfracs == 1:
                        ydots[specie] += -r.eval(T, composition, density) * Counter(r.reactants)[specie] * \
                                          functools.reduce(mul, [composition[q]*density/self.sympy_to_python(q).m for q in r.reactants])

                    elif self.massfracs == 0:
                        ydots[specie] += -r.eval(T, composition, density) * Counter(r.reactants)[specie] * \
                                          functools.reduce(mul, [composition[q] for q in r.reactants])

                if specie in r.products:

                    if self.massfracs == 1:
                        ydots[specie] += r.eval(T, composition, density) * Counter(r.products)[specie] * \
                                         functools.reduce(mul, [composition[q]*density/self.sympy_to_python(q).m for q in r.reactants])

                    elif self.massfracs == 0:
                        ydots[specie] += r.eval(T, composition, density) * Counter(r.products)[specie] * \
                                         functools.reduce(mul, [composition[q] for q in r.reactants])

        return ydots

    def evaluate_cooling(self, T, composition, redshift, density=0):
        #NOTE - CIE cooling not used as per krome's test
        sumcool = self.evaluate_cooling_cont(T, composition, density) + self.evaluate_cooling_compton(T, composition, redshift, density) + \
                  self.evaluate_cooling_chem(T, composition, density) + self.evaluate_cooling_atomic(T, composition, density) + \
                  self.evaluate_cooling_H2(T, composition, density)

        if self.withD == 1:
            sumcool += self.evaluate_cooling_HD(T, composition, redshift, density)

        return sumcool

    def evaluate_heating(self, T, composition, redshift, density=0):
        sumheat = self.evaluate_heating_chem(T, composition, density) + self.evaluate_heating_compress(T, composition, density)

        return sumheat

    def evaluate_tdot(self, T, composition, redshift, density=0):
        if self.tdot_switch == 0:
            #dT/ dt
            tdot = (self.evaluate_heating(T, composition, redshift, density) - self.evaluate_cooling(T, composition, redshift, density)) * \
                   (self.evaluate_gamma(composition, density) - 1.0) / cons.boltzmann_erg / self.get_n(composition, density)

        elif self.tdot_switch == 1:
            #dEint / dt
            if density == 0:
                density = self.get_rho(composition)

            tdot = (self.evaluate_heating(T, composition, redshift, density) - self.evaluate_cooling(T, composition, redshift, density)) / density

        return tdot
    
    def get_Hnuclei(self, composition, density=0):
        import sympy as sp

        if self.massfracs == 1:
            nH = composition[ChemSpecie('hp').sym_name]/ChemSpecie('hp').m + composition[ChemSpecie('h').sym_name]/ChemSpecie('h').m + composition[ChemSpecie('hm').sym_name]/ChemSpecie('hm').m + \
                 composition[ChemSpecie('h2').sym_name]*2.0/ChemSpecie('h2').m + composition[ChemSpecie('h2p').sym_name]*2.0/ChemSpecie('h2p').m

            if self.withD == 1:
                nH += composition[ChemSpecie('hd').sym_name]/ChemSpecie('hd').m + composition[ChemSpecie('hdp').sym_name]/ChemSpecie('hdp').m

            return nH*density

        elif self.massfracs == 0:
            nH = composition[ChemSpecie('hp').sym_name] + composition[ChemSpecie('h').sym_name] + composition[ChemSpecie('hm').sym_name] + \
                 composition[ChemSpecie('h2').sym_name]*2.0 + composition[ChemSpecie('h2p').sym_name]*2.0

            if self.withD == 1:
                nH += composition[ChemSpecie('hd').sym_name] + composition[ChemSpecie('hdp').sym_name]

            return nH

    def get_n(self, composition, density=0):

        if self.massfracs == 1:
            n = composition[ChemSpecie('hp').sym_name]/ChemSpecie('hp').m + composition[ChemSpecie('h').sym_name]/ChemSpecie('h').m + composition[ChemSpecie('hm').sym_name]/ChemSpecie('hm').m + \
                composition[ChemSpecie('h2').sym_name]/ChemSpecie('h2').m + composition[ChemSpecie('h2p').sym_name]/ChemSpecie('h2p').m + composition[ChemSpecie('hepp').sym_name]/ChemSpecie('hepp').m + \
                composition[ChemSpecie('he').sym_name]/ChemSpecie('he').m + composition[ChemSpecie('hep').sym_name]/ChemSpecie('hep').m + composition[ChemSpecie('elec').sym_name]/ChemSpecie('elec').m
                
            if self.withD == 1:
                n += composition[ChemSpecie('hd').sym_name]/ChemSpecie('hd').m + composition[ChemSpecie('hdp').sym_name]/ChemSpecie('hdp').m + \
                     composition[ChemSpecie('dm').sym_name]/ChemSpecie('dm').m + composition[ChemSpecie('d').sym_name]/ChemSpecie('d').m + composition[ChemSpecie('dp').sym_name]/ChemSpecie('dp').m

            return n*density

        elif self.massfracs == 0:
            n = composition[ChemSpecie('hp').sym_name] + composition[ChemSpecie('h').sym_name] + composition[ChemSpecie('hm').sym_name] + \
                composition[ChemSpecie('h2').sym_name] + composition[ChemSpecie('h2p').sym_name] + composition[ChemSpecie('hepp').sym_name] + \
                composition[ChemSpecie('he').sym_name] + composition[ChemSpecie('hep').sym_name] + composition[ChemSpecie('elec').sym_name]
                
            if self.withD == 1:
                n += composition[ChemSpecie('hd').sym_name] + composition[ChemSpecie('hdp').sym_name] + \
                     composition[ChemSpecie('dm').sym_name] + composition[ChemSpecie('d').sym_name] + composition[ChemSpecie('dp').sym_name]

            return n


    def get_rho(self, composition, density=0):

        if self.massfracs == 1:
            return density

        elif self.massfracs == 0:
            rho = composition[ChemSpecie('hp').sym_name]*ChemSpecie('hp').m + composition[ChemSpecie('h').sym_name]*ChemSpecie('h').m + \
                  composition[ChemSpecie('hm').sym_name]*ChemSpecie('hm').m + composition[ChemSpecie('h2').sym_name]*ChemSpecie('h2').m + \
                  composition[ChemSpecie('h2p').sym_name]*ChemSpecie('h2p').m + composition[ChemSpecie('he').sym_name]*ChemSpecie('he').m + \
                  composition[ChemSpecie('hep').sym_name]*ChemSpecie('hep').m + composition[ChemSpecie('hepp').sym_name]*ChemSpecie('hepp').m + \
                  composition[ChemSpecie('elec').sym_name]*ChemSpecie('elec').m             

            if self.withD == 1:
                rho += composition[ChemSpecie('hd').sym_name]*ChemSpecie('hd').m + composition[ChemSpecie('hdp').sym_name]*ChemSpecie('hdp').m + \
                       composition[ChemSpecie('d').sym_name]*ChemSpecie('d').m + composition[ChemSpecie('dm').sym_name]*ChemSpecie('dm').m + \
                       composition[ChemSpecie('dp').sym_name]*ChemSpecie('dp').m
    
            return rho


    def evaluate_heating_chem(self, T, composition, density=0):
        import sympy as sp

        dd = self.get_Hnuclei(composition, density)
        small = 1e-99
        heatingChem = 0.

        ncrn  = 1.0e6*(T**(-0.5))
        ncrd1 = 1.6*sp.exp(-(4.0e2/T)**2)
        ncrd2 = 1.4*sp.exp(-1.2e4/(T+1.2e3))

        if self.massfracs == 1:
            yH = composition[ChemSpecie('h').sym_name]*density/(dd*ChemSpecie('h').m)
            yH2 = composition[ChemSpecie('h2').sym_name]*density/(dd*ChemSpecie('h2').m)
        elif self.massfracs == 0:
            yH = composition[ChemSpecie('h').sym_name]
            yH2 = composition[ChemSpecie('h2').sym_name]

        ncr = ncrn/(ncrd1*yH + ncrd2*yH2)
        h2heatfac = 1.0/(1.0 + ncr/dd)

        HChem = 0.
        a1, b1, c1, d1 = 0., 0., 0., 0.

        for r in self.rates:
            if {ChemSpecie('hm').sym_name, ChemSpecie('h').sym_name} == set(r.reactants) and {ChemSpecie('h2').sym_name, ChemSpecie('elec').sym_name} == set(r.products):
                #reaction 10
                if self.massfracs == 1:
                    a1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (3.53*h2heatfac*functools.reduce(mul, [composition[q]*density/self.sympy_to_python(q).m for q in r.reactants]))
                elif self.massfracs == 0:
                    a1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (3.53*h2heatfac*functools.reduce(mul, [composition[q] for q in r.reactants]))


        for r in self.rates:
            if {ChemSpecie('h2p').sym_name, ChemSpecie('h').sym_name} == set(r.reactants) and {ChemSpecie('hp').sym_name, ChemSpecie('h2').sym_name} == set(r.products):
                #reaction 13
                if self.massfracs == 1:
                    b1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (1.83*h2heatfac*functools.reduce(mul, [composition[q]*density/self.sympy_to_python(q).m for q in r.reactants]))
                elif self.massfracs == 0:
                    b1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (1.83*h2heatfac*functools.reduce(mul, [composition[q] for q in r.reactants]))

        for r in self.rates:
            if Counter(r.reactants)[ChemSpecie('h').sym_name] == 3 and {ChemSpecie('h2').sym_name, ChemSpecie('h').sym_name} == set(r.products):
                #reaction 25
                if self.massfracs == 1:
                    c1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (4.48*h2heatfac*functools.reduce(mul, [composition[q]*density/self.sympy_to_python(q).m for q in r.reactants]))
                elif self.massfracs == 0:
                    c1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (4.48*h2heatfac*functools.reduce(mul, [composition[q] for q in r.reactants]))

        for r in self.rates:
            if Counter(r.reactants)[ChemSpecie('h').sym_name] == 2 and Counter(r.reactants)[ChemSpecie('h2').sym_name] == 1 and Counter(r.products)[ChemSpecie('h2').sym_name] == 2:
                #reaction 26
                if self.massfracs == 1:
                    d1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (4.48*h2heatfac*functools.reduce(mul, [composition[q]*density/self.sympy_to_python(q).m for q in r.reactants]))
                elif self.massfracs == 0:
                    d1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (4.48*h2heatfac*functools.reduce(mul, [composition[q] for q in r.reactants]))

        HChem = a1 + b1 + c1 + d1
        return HChem*cons.eV_to_erg

    def get_free_fall_time(self, composition, density=0):
        import sympy as sp
        rhogas = self.get_rho(composition, density)
        free_fall_time = sp.sqrt(3.0*sp.pi/32.0/cons.gravity/rhogas)
        return free_fall_time

    def evaluate_heating_compress(self, T, composition, density=0):

        free_fall_time = self.get_free_fall_time(composition, density)

        dd = self.get_n(composition, density)

        Hcompress = dd * cons.boltzmann_erg * T / free_fall_time #erg/s/cm3
        return Hcompress


    def evaluate_cooling_chem(self, T, composition, density=0):

        CChem = 0.
        a1, b1, c1 = 0., 0., 0.

        for r in self.rates:
            if {ChemSpecie('h2').sym_name, ChemSpecie('elec').sym_name} == set(r.reactants) and Counter(r.products)[ChemSpecie('h').sym_name] == 2 and Counter(r.products)[ChemSpecie('elec').sym_name] == 1:
                #reaction 16
                if self.massfracs == 1:
                    a1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (4.48*functools.reduce(mul, [composition[q]*density/self.sympy_to_python(q).m for q in r.reactants]))
                elif self.massfracs == 0:
                    a1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (4.48*functools.reduce(mul, [composition[q] for q in r.reactants]))

        for r in self.rates:
            if {ChemSpecie('h2').sym_name, ChemSpecie('h').sym_name} == set(r.reactants) and Counter(r.products)[ChemSpecie('h').sym_name] == 3:
                #reaction 17
                if self.massfracs == 1:
                    b1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (4.48*functools.reduce(mul, [composition[q]*density/self.sympy_to_python(q).m for q in r.reactants]))
                elif self.massfracs == 0:
                    b1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (4.48*functools.reduce(mul, [composition[q] for q in r.reactants]))

        for r in self.rates:
            if Counter(r.reactants)[ChemSpecie('h2').sym_name] == 2 and Counter(r.products)[ChemSpecie('h').sym_name] == 2 and Counter(r.products)[ChemSpecie('h2').sym_name] == 1:
                #reaction 27
                if self.massfracs == 1:
                    c1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (4.48*functools.reduce(mul, [composition[q]*density/self.sympy_to_python(q).m for q in r.reactants]))
                elif self.massfracs == 0:
                    c1 = SympyChemRate(reactants=r.reactants, products=r.products).eval(T, composition, density) * \
                             (4.48*functools.reduce(mul, [composition[q] for q in r.reactants]))

        CChem = a1 + b1 + c1

        return CChem*cons.eV_to_erg       
    
    def evaluate_cooling_atomic(self, T, composition, density=0):
        import sympy as sp
        temp_old = sp.Max(T,10) #K
        #if you dont rewrite dd_old as below, it will throw an error while using cxxcode on the derivative of dd_old later on
        #because cxxcode does not like heaviside function which will appear in the derivative of dd_old
        temp = temp_old.rewrite(sp.Piecewise)

        T5 = temp/1e5 
        
        Catomic = 0.

        if self.massfracs == 1:

            #COLLISIONAL IONIZATION: H, He, He+, He(2S)
            Catomic = Catomic+ 1.27e-21*sp.sqrt(temp)/(1.0+sp.sqrt(T5))*sp.exp(-1.578091e5/temp)*composition[ChemSpecie('elec').sym_name]*(density/ChemSpecie('elec').m) * \
                      composition[ChemSpecie('h').sym_name]*(density/ChemSpecie('h').m)

            Catomic = Catomic + 9.38e-22*sp.sqrt(temp)/(1.0+sp.sqrt(T5))*sp.exp(-2.853354e5/temp)*composition[ChemSpecie('elec').sym_name]*(density/ChemSpecie('elec').m) * \
                      composition[ChemSpecie('he').sym_name]*(density/ChemSpecie('he').m)

            Catomic = Catomic + 4.95e-22*sp.sqrt(temp)/(1.0+sp.sqrt(T5))*sp.exp(-6.31515e5/temp)*composition[ChemSpecie('elec').sym_name]*(density/ChemSpecie('elec').m) * \
                      composition[ChemSpecie('hep').sym_name]*(density/ChemSpecie('hep').m)
            Catomic = Catomic + 5.01e-27*temp**(-0.1687)/(1.0+sp.sqrt(T5))*sp.exp(-5.5338e4/temp)*(composition[ChemSpecie('elec').sym_name]*density/ChemSpecie('elec').m)**2 * \
                      composition[ChemSpecie('hep').sym_name]*(density/ChemSpecie('hep').m)


            #RECOMBINATION: H+, He+,He2+
            Catomic = Catomic + 8.7e-27*sp.sqrt(temp)*(temp/1.e3)**(-0.2)/(1.0+(temp/1.e6)**0.7)*composition[ChemSpecie('elec').sym_name]*(density/ChemSpecie('elec').m) * \
                      composition[ChemSpecie('hp').sym_name]*(density/ChemSpecie('hp').m)

            Catomic = Catomic + 1.55e-26*temp**(0.3647)*composition[ChemSpecie('elec').sym_name]*(density/ChemSpecie('elec').m)*composition[ChemSpecie('hep').sym_name]*(density/ChemSpecie('hep').m)

            Catomic = Catomic + 3.48e-26*sp.sqrt(temp)*(temp/1.e3)**(-0.2)/(1.0+(temp/1.e6)**0.7)*composition[ChemSpecie('elec').sym_name]*(density/ChemSpecie('elec').m) * \
                      composition[ChemSpecie('hepp').sym_name]*(density/ChemSpecie('hepp').m)


            #!DIELECTRONIC RECOMBINATION: He
            Catomic = Catomic + 1.24e-13*temp**(-1.5)*sp.exp(-4.7e5/temp)*(1.0+0.30*sp.exp(-9.4e4/temp))*composition[ChemSpecie('elec').sym_name]*(density/ChemSpecie('elec').m) * \
                      composition[ChemSpecie('hep').sym_name]*(density/ChemSpecie('hep').m)


            #COLLISIONAL EXCITATION:
            #H(all n), He(n=2,3,4 triplets), He+(n=2)
            Catomic = Catomic + 7.5e-19/(1.0+sp.sqrt(T5))*sp.exp(-1.18348e5/temp)*composition[ChemSpecie('elec').sym_name]*(density/ChemSpecie('elec').m) * \
                      composition[ChemSpecie('h').sym_name]*(density/ChemSpecie('h').m)

            Catomic = Catomic + 9.1e-27*temp**(-.1687)/(1.0+sp.sqrt(T5))*sp.exp(-1.3179e4/temp)*(composition[ChemSpecie('elec').sym_name]*density/ChemSpecie('elec').m)**2 * \
                      composition[ChemSpecie('hep').sym_name]*(density/ChemSpecie('hep').m)
            Catomic = Catomic + 5.54e-17*temp**(-.397)/(1.0+sp.sqrt(T5))*sp.exp(-4.73638e5/temp)*composition[ChemSpecie('elec').sym_name]*(density/ChemSpecie('elec').m) * \
                      composition[ChemSpecie('hep').sym_name]*(density/ChemSpecie('hep').m)

        elif self.massfracs == 0:

            #COLLISIONAL IONIZATION: H, He, He+, He(2S)
            Catomic = Catomic+ 1.27e-21*sp.sqrt(temp)/(1.0+sp.sqrt(T5))*sp.exp(-1.578091e5/temp)*composition[ChemSpecie('elec').sym_name] * \
                      composition[ChemSpecie('h').sym_name]

            Catomic = Catomic + 9.38e-22*sp.sqrt(temp)/(1.0+sp.sqrt(T5))*sp.exp(-2.853354e5/temp)*composition[ChemSpecie('elec').sym_name] * \
                      composition[ChemSpecie('he').sym_name]

            Catomic = Catomic + 4.95e-22*sp.sqrt(temp)/(1.0+sp.sqrt(T5))*sp.exp(-6.31515e5/temp)*composition[ChemSpecie('elec').sym_name] * \
                      composition[ChemSpecie('hep').sym_name]
            Catomic = Catomic + 5.01e-27*temp**(-0.1687)/(1.0+sp.sqrt(T5))*sp.exp(-5.5338e4/temp)*(composition[ChemSpecie('elec').sym_name])**2 * \
                      composition[ChemSpecie('hep').sym_name]


            #RECOMBINATION: H+, He+,He2+
            Catomic = Catomic + 8.7e-27*sp.sqrt(temp)*(temp/1.e3)**(-0.2)/(1.0+(temp/1.e6)**0.7)*composition[ChemSpecie('elec').sym_name] * \
                      composition[ChemSpecie('hp').sym_name]

            Catomic = Catomic + 1.55e-26*temp**(0.3647)*composition[ChemSpecie('elec').sym_name]*composition[ChemSpecie('hep').sym_name]

            Catomic = Catomic + 3.48e-26*sp.sqrt(temp)*(temp/1.e3)**(-0.2)/(1.0+(temp/1.e6)**0.7)*composition[ChemSpecie('elec').sym_name] * \
                      composition[ChemSpecie('hepp').sym_name]


            #!DIELECTRONIC RECOMBINATION: He
            Catomic = Catomic + 1.24e-13*temp**(-1.5)*sp.exp(-4.7e5/temp)*(1.0+0.30*sp.exp(-9.4e4/temp))*composition[ChemSpecie('elec').sym_name] * \
                      composition[ChemSpecie('hep').sym_name]


            #COLLISIONAL EXCITATION:
            #H(all n), He(n=2,3,4 triplets), He+(n=2)
            Catomic = Catomic + 7.5e-19/(1.0+sp.sqrt(T5))*sp.exp(-1.18348e5/temp)*composition[ChemSpecie('elec').sym_name] * \
                      composition[ChemSpecie('h').sym_name]

            Catomic = Catomic + 9.1e-27*temp**(-.1687)/(1.0+sp.sqrt(T5))*sp.exp(-1.3179e4/temp)*(composition[ChemSpecie('elec').sym_name])**2 * \
                      composition[ChemSpecie('hep').sym_name]
            Catomic = Catomic + 5.54e-17*temp**(-.397)/(1.0+sp.sqrt(T5))*sp.exp(-4.73638e5/temp)*composition[ChemSpecie('elec').sym_name] * \
                      composition[ChemSpecie('hep').sym_name]


        return Catomic

    def evaluate_cooling_compton(self, T, composition, redshift, density=0):
        if self.massfracs == 1:
            Ccompton = 5.65e-36 * (1.0 + redshift)**4 * (T - 2.73 * (1.0 + redshift)) * composition[ChemSpecie('elec').sym_name]*(density/ChemSpecie('elec').m) #erg/s/cm3
        elif self.massfracs == 0:
            Ccompton = 5.65e-36 * (1.0 + redshift)**4 * (T - 2.73 * (1.0 + redshift)) * composition[ChemSpecie('elec').sym_name] #erg/s/cm3
        return Ccompton

    def kpla(self, composition, density=0):
        import sympy as sp
        rhogas = self.get_rho(composition, density)

        kpla = 0.0
        
        a0 = 1.000042e0
        a1 = 2.14989e0

        #log density cannot exceed 0.5 g/cm3
        y_old = sp.log(sp.Min(rhogas,0.50), 10)
        y = y_old.rewrite(sp.Piecewise)

        kpla = 1e1**(a0*y + a1) #fit density only

        #opacity is zero under 1e-12 g/cm3
        final = sp.Piecewise((kpla, rhogas >= 1e-12), (0., rhogas < 1e-12))

        return final

    def get_jeans_length(self, T, composition, density=0):
        import sympy as sp
        rhogas_old = sp.Max(self.get_rho(composition, density), 1e-40)
        rhogas = rhogas_old.rewrite(sp.Piecewise)
        mu_old = rhogas / sp.Max(self.get_n(composition, density), 1e-40) * cons.ip_mass
        mu = mu_old.rewrite(sp.Piecewise)
        get_jeans_length = sp.sqrt(sp.pi * cons.boltzmann_erg * T/rhogas / cons.p_mass / cons.gravity / mu)
        return get_jeans_length

    def evaluate_cooling_cont(self, T, composition, density=0):
        import sympy as sp
        rhogas = self.get_rho(composition, density) #g/cm3
        kgas = self.kpla(composition, density) #planck opacity cm2/g (Omukai+2000)
        lj = self.get_jeans_length(T, composition, density) #cm
        tau = lj * kgas * rhogas + 1e-40 #opacity
        beta_old = sp.Min(1.0, tau**(-2)) #beta escape (always <1.)
        beta = beta_old.rewrite(sp.Piecewise)
        Ccont = 4.0 * cons.stefboltz_erg * (T**4) * kgas * rhogas * beta #erg/s/cm3
        return Ccont

    def evaluate_cooling_CIE(self, T, composition, redshift, density=0):
        import sympy as sp
        CCIE = 0.0
        Tcmb = 2.73*(1+redshift)
        #set cooling to zero if n_H2 is smaller than 1e-12 1/cm3
        #to avoid division by zero in opacity term due to tauCIE=0
        #if composition[ChemSpecie('h2').sym_name] < 1e-12:
        #    return 0.

        #if T < Tcmb:
        #    return 0.

        x = sp.log(T, 10)
        x2 = x*x
        x3 = x2*x
        x4 = x3*x
        x5 = x4*x

        cool = 0.0

        a0 = -30.3314216559651
        a1 = 19.0004016698518
        a2 = -17.1507937874082
        a3 = 9.49499574218739
        a4 = -2.54768404538229
        a5 = 0.265382965410969
        expr1 = a0 + a1*x + a2*x2 + a3*x3 +a4*x4 +a5*x5

        b0 = -180.992524120965
        b1 = 168.471004362887
        b2 = -67.499549702687
        b3 = 13.5075841245848
        b4 = -1.31983368963974
        b5 = 0.0500087685129987
        expr2 = b0 + b1*x + b2*x2 + b3*x3 +b4*x4 +b5*x5

        expr3 = 3.0 * x - 21.2968837223113 #cubic extrapolation

        #outside boundaries below cooling is zero
        logcool = sp.Piecewise((-1e99, T <= 2), (expr1, sp.And(T> 2, T < 2.95)), (expr2, sp.And(T>=2.95, T<5)), (expr3, T>=5))

        #opacity according to RA04
        if self.massfracs == 1:
            tauCIE = (composition[ChemSpecie('h2').sym_name]*(density/ChemSpecie('h2').m) * 1.4285714e-16)**2.8 #note: 1/7d15 = 1.4285714d-16
        elif self.massfracs == 0:
            tauCIE = (composition[ChemSpecie('h2').sym_name] * 1.4285714e-16)**2.8 #note: 1/7d15 = 1.4285714d-16
        cool = cons.p_mass * 1e1**logcool #erg*cm3/s

        if self.massfracs == 1:
            expr4_old = cool * sp.Min(1.0, (1.0-sp.exp(-tauCIE))/tauCIE) * composition[ChemSpecie('h2').sym_name]*(density/ChemSpecie('h2').m) * self.get_n(composition, density) #erg/cm3/s
        elif self.massfracs == 0:
            expr4_old = cool * sp.Min(1.0, (1.0-sp.exp(-tauCIE))/tauCIE) * composition[ChemSpecie('h2').sym_name] * self.get_n(composition, density) #erg/cm3/s
        
        expr4 = expr4_old.rewrite(sp.Piecewise)

        if self.massfracs == 1:
            CCIE = sp.Piecewise((expr4, sp.Or(composition[ChemSpecie('h2').sym_name]*density/ChemSpecie('h2').m >= 1e-12, T >= Tcmb)), (0, True))
        elif self.massfracs == 0:
            CCIE = sp.Piecewise((expr4, sp.Or(composition[ChemSpecie('h2').sym_name] >= 1e-12, T >= Tcmb)), (0, True))

        return CCIE

    def evaluate_cooling_HD(self, T, composition, redshift, density=0):
        import sympy as sp
        Tcmb = 2.73*(1+redshift)
        CHD = 0.0 #erg/cm3/s

        #this function does not have limits on density
        #and temperature, even if the original paper do.
        #However, we extrapolate the limits.

        #exit on low temperature
        #if(T < Tcmb):
        #    return 0.0

        #extrapolate higher temperature limit
        Tgas_old = sp.Min(T,1e4)
        Tgas = Tgas_old.rewrite(sp.Piecewise)

        #calculate density
        if self.massfracs == 1:
            dd = composition[ChemSpecie('h').sym_name]*density/ChemSpecie('h').m #self.get_n(composition)
        elif self.massfracs == 0:
            dd = composition[ChemSpecie('h').sym_name] #self.get_n(composition)
        #exit if density is out of Lipovka bounds (uncomment if needed)
        #if(dd<1d0 .or. dd>1d8) return
        #extrapolate density limits
        dd_old = sp.Min(sp.Max(dd,1e-2),1e10)
        #if you dont rewrite dd_old as below, it will throw an error while using cxxcode on the derivative of dd_old later on
        #because cxxcode does not like heaviside function which will appear in the derivative of dd_old
        dd = dd_old.rewrite(sp.Piecewise)

        #POLYNOMIAL COEFFICIENT: TABLE 1 LIPOVKA
        lipovka = np.zeros((5,5))
        lipovka[0,:] = np.array([-42.56788, 0.92433, 0.54962, -0.07676, 0.00275])
        lipovka[1,:] = np.array([21.93385, 0.77952, -1.06447, 0.11864, -0.00366])
        lipovka[2,:] = np.array([-10.19097, -0.54263, 0.62343, -0.07366, 0.002514])
        lipovka[3,:] = np.array([2.19906, 0.11711, -0.13768, 0.01759, -0.00066631])
        lipovka[4,:] = np.array([-0.17334, -0.00835, 0.0106, -0.001482, 0.00006192])

        logTgas = sp.log(Tgas, 10)
        lognH   = sp.log(dd, 10)

        #loop to compute coefficients
        logW = 0.0
        for j in range(0,5):
            lHj = lognH**j
            for i in range(0,5):
                logW += lipovka[i,j]*logTgas**i*lHj #erg/s

        W = 10.0**(logW)

        if self.massfracs == 1:
            expr = W * composition[ChemSpecie('hd').sym_name]*density/ChemSpecie('hd').m #erg/cm3/s
        elif self.massfracs == 0:
            expr = W * composition[ChemSpecie('hd').sym_name] #erg/cm3/s

        CHD = sp.Piecewise((expr, T>=Tcmb), (0, True))
        return CHD


    def sigmoid(self, x, x0, s):
        import sympy as sp
        sigmoid = 1e1/(1e1+sp.exp(-s*(x-x0)))
        return sigmoid

    def wCool(self, logTgas, logTmin, logTmax):
        import sympy as sp
        x = (logTgas-logTmin)/(logTmax-logTmin)

        wCool = 1e1**(2e2*(self.sigmoid(x,-2e-1,5e1)*self.sigmoid(-x,-1.2,5e1)-1.0))

        #if wCool < 1e-199:
        #    wCool = 0.0
        
        #NOT SURE HOW TO SYMPY THIS
        #HARDCODED
        #if wCool > 1:
        #    raise ValueError("wCool > 1 in H2 cooling!")

        return wCool

    def evaluate_cooling_H2(self, T, composition, density=0):
        import sympy as sp
        temp = T
        CH2 = 0.0
        #if(temp<2d0) return

        t3 = temp * 1e-3
        logt3 = sp.log(t3, 10)
        logt = sp.log(temp, 10)
        cool = 0.0

        logt32 = logt3 * logt3
        logt33 = logt32 * logt3
        logt34 = logt33 * logt3
        logt35 = logt34 * logt3
        logt36 = logt35 * logt3
        logt37 = logt36 * logt3
        logt38 = logt37 * logt3

        w14 = self.wCool(logt, 1.0, 4.0)
        w24 = self.wCool(logt, 2.0, 4.0)

        if self.massfracs == 1:
            #//H2-H
            expr1 = 1.e1**(-16.818342 + 3.7383713e1*logt3 + 5.8145166e1*logt32 + \
                           4.8656103e1*logt33 + 2.0159831e1*logt34 + 3.8479610*logt35) * composition[ChemSpecie('h').sym_name]*density/ChemSpecie('h').m
            
            expr2 = 1.e1**(-2.4311209e1 + 3.5692468*logt3 -1.1332860e1*logt32 - 2.7850082e1*logt33 - \
                           2.1328264e1*logt34 - 4.2519023*logt35) * composition[ChemSpecie('h').sym_name]*density/ChemSpecie('h').m

            expr3 = 1e1**(-2.4311209e1 + 4.6450521*logt3 - 3.7209846*logt32 + 5.9369081*logt33 - \
                          5.5108049*logt34 + 1.5538288*logt35) * composition[ChemSpecie('h').sym_name]*density/ChemSpecie('h').m

            expr4 = 1.862314467912518e-022*self.wCool(logt,1.0,np.log10(6e3)) * composition[ChemSpecie('h').sym_name]*density/ChemSpecie('h').m

            fH2H = sp.Piecewise((expr1, temp <=1e2), (expr2, sp.And(temp >1e2, temp <=1e3)), (expr3, sp.And(temp > 1e3, temp <= 6e3)), (expr4, temp > 6e3))

            cool += fH2H

            #//H2-Hp
            expr5 = 1e1**(-2.2089523e1 +1.5714711*logt3 + 0.015391166*logt32 - 0.23619985*logt33 - \
                           0.51002221*logt34 + 0.32168730*logt35) * composition[ChemSpecie('hp').sym_name]*density/ChemSpecie('hp').m
            
            expr6 = 1.182509139382060e-021 * composition[ChemSpecie('hp').sym_name]*(density/ChemSpecie('hp').m) * w14

            fH2Hp = sp.Piecewise((expr5, sp.And(temp > 1e1, temp <=1e4)), (expr6, True))

            cool += fH2Hp

            #//H2-H2
            fH2H2 = w24 * 1e1**(-2.3962112e1 + 2.09433740*logt3 - 0.77151436*logt32 + \
                                 0.43693353*logt33 - 0.14913216*logt34 - 0.033638326*logt35) * composition[ChemSpecie('h2').sym_name]*density/ChemSpecie('h2').m

            cool += fH2H2

            #//H2-e
            #placed lower limit of 100K as reported in the original data by Glover 2015, MNRAS 451
            #this lower limit was not adopted by KROME developers
            expr7 = 1e1**(-2.1928796e1 + 1.6815730e1*logt3 + 9.6743155e1*logt32 + 3.4319180e2*logt33 + \
                              7.3471651e2*logt34 + 9.8367576e2*logt35 + 8.0181247e2*logt36 + \
                              3.6414446e2*logt37 + 7.0609154e1*logt38) * composition[ChemSpecie('elec').sym_name]*density/ChemSpecie('elec').m

            expr8 = 1e1**(-2.2921189e1 + 1.6802758*logt3 + 0.93310622*logt32 + 4.0406627*logt33 - \
                          4.7274036*logt34 - 8.8077017*logt35 + 8.9167183*logt36 + 6.4380698*logt37 - \
                          6.3701156*logt38) * composition[ChemSpecie('elec').sym_name]*density/ChemSpecie('elec').m

            fH2e = sp.Piecewise((expr7, sp.And(temp > 100, temp <=5e2)), (expr8, temp > 5e2), (0, True))    

            cool += fH2e*w24

            #//H2-He
            expr9 = 1e1**(-2.3689237e1 +2.1892372*logt3 - 0.81520438*logt32 + 0.29036281*logt33 - \
                           0.16596184*logt34 + 0.19191375*logt35) * composition[ChemSpecie('he').sym_name]*density/ChemSpecie('he').m

            expr10 = 1.002560385050777e-022 * composition[ChemSpecie('he').sym_name]*(density/ChemSpecie('he').m) * w14

            fH2He = sp.Piecewise((expr9, sp.And(temp > 1e1, temp <=1e4)), (expr10, True))
            cool += fH2He


        elif self.massfracs == 0:
            #//H2-H
            expr1 = 1.e1**(-16.818342 + 3.7383713e1*logt3 + 5.8145166e1*logt32 + \
                           4.8656103e1*logt33 + 2.0159831e1*logt34 + 3.8479610*logt35) * composition[ChemSpecie('h').sym_name]
            
            expr2 = 1.e1**(-2.4311209e1 + 3.5692468*logt3 -1.1332860e1*logt32 - 2.7850082e1*logt33 - \
                           2.1328264e1*logt34 - 4.2519023*logt35) * composition[ChemSpecie('h').sym_name]

            expr3 = 1e1**(-2.4311209e1 + 4.6450521*logt3 - 3.7209846*logt32 + 5.9369081*logt33 - \
                          5.5108049*logt34 + 1.5538288*logt35) * composition[ChemSpecie('h').sym_name]

            expr4 = 1.862314467912518e-022*self.wCool(logt,1.0,np.log10(6e3)) * composition[ChemSpecie('h').sym_name]

            fH2H = sp.Piecewise((expr1, temp <=1e2), (expr2, sp.And(temp >1e2, temp <=1e3)), (expr3, sp.And(temp > 1e3, temp <= 6e3)), (expr4, temp > 6e3))

            cool += fH2H

            #//H2-Hp
            expr5 = 1e1**(-2.2089523e1 +1.5714711*logt3 + 0.015391166*logt32 - 0.23619985*logt33 - \
                           0.51002221*logt34 + 0.32168730*logt35) * composition[ChemSpecie('hp').sym_name]
            
            expr6 = 1.182509139382060e-021 * composition[ChemSpecie('hp').sym_name] * w14

            fH2Hp = sp.Piecewise((expr5, sp.And(temp > 1e1, temp <=1e4)), (expr6, True))

            cool += fH2Hp

            #//H2-H2
            fH2H2 = w24 * 1e1**(-2.3962112e1 + 2.09433740*logt3 - 0.77151436*logt32 + \
                                 0.43693353*logt33 - 0.14913216*logt34 - 0.033638326*logt35) * composition[ChemSpecie('h2').sym_name]

            cool += fH2H2

            #//H2-e
            #placed lower limit of 100K as reported in the original data by Glover 2015, MNRAS 451
            #this lower limit was not adopted by KROME developers
            expr7 = 1e1**(-2.1928796e1 + 1.6815730e1*logt3 + 9.6743155e1*logt32 + 3.4319180e2*logt33 + \
                              7.3471651e2*logt34 + 9.8367576e2*logt35 + 8.0181247e2*logt36 + \
                              3.6414446e2*logt37 + 7.0609154e1*logt38) * composition[ChemSpecie('elec').sym_name]

            expr8 = 1e1**(-2.2921189e1 + 1.6802758*logt3 + 0.93310622*logt32 + 4.0406627*logt33 - \
                          4.7274036*logt34 - 8.8077017*logt35 + 8.9167183*logt36 + 6.4380698*logt37 - \
                          6.3701156*logt38) * composition[ChemSpecie('elec').sym_name]

            fH2e = sp.Piecewise((expr7, sp.And(temp > 100, temp <=5e2)), (expr8, temp > 5e2), (0, True))    

            cool += fH2e*w24

            #//H2-He
            expr9 = 1e1**(-2.3689237e1 +2.1892372*logt3 - 0.81520438*logt32 + 0.29036281*logt33 - \
                           0.16596184*logt34 + 0.19191375*logt35) * composition[ChemSpecie('he').sym_name]

            expr10 = 1.002560385050777e-022 * composition[ChemSpecie('he').sym_name] * w14

            fH2He = sp.Piecewise((expr9, sp.And(temp > 1e1, temp <=1e4)), (expr10, True))
            cool += fH2He

        #check error
        #DONT KNOW HOW TO SYMPY THIS
        #HARDCODED
        #if cool > 1e30:
        #    raise ValueError(" ERROR: cooling >1.d30 erg/s/cm3")

        #this to avoid negative, overflow and useless calculations below
        #if cool <= 0.0:
        #    CH2 = 0.0
        #    return CH2

        #high density limit from HM79, GP98 below Tgas = 2d3
        #UPDATED USING GLOVER 2015 for high temperature corrections, MNRAS
        #IN THE HIGH DENSITY REGIME LAMBDA_H2 = LAMBDA_H2(LTE) = HDL
        #the following mix of functions ensures the right behaviour
        # at low (T<10 K) and high temperatures (T>2000 K) by
        # using both the original Hollenbach and the new Glover data
        # merged in a smooth way.


        #HIGH DENSITY COOLING GIVING PROBLEMS WITH SYMPY. 
        #To avoid problems with lambdifying using numpy, I defined 'factor' below, and then lambdify using math

        HDLR = ((9.5e-22*t3**3.76)/(1.+0.12*t3**2.1)*sp.exp(-(0.13/t3)**3) + 3.e-24*sp.exp(-0.51/t3)) #erg/s
        HDLV = (6.7e-19*sp.exp(-5.86/t3) + 1.6e-18*sp.exp(-11.7/t3)) #erg/s
        expr11  = HDLR + HDLV #erg/s

        expr12 = 1e1**(-2.0584225e1 + 5.0194035*logt3 - 1.5738805*logt32 - 4.7155769*logt33 + \
                     2.4714161*logt34 + 5.4710750*logt35 - 3.9467356*logt36 - 2.2148338*logt37 + 1.8161874*logt38)

        dump14_old = 1.0 / (1.0 + sp.exp(sp.Min((temp-3e4)*2e-4,3e2)))
        dump14 = dump14_old.rewrite(sp.Piecewise)

        expr13 = 5.531333679406485E-019*dump14

        HDL = sp.Piecewise((expr11, temp < 2e3), (expr12, sp.And(temp >= 2e3, temp <=1e4)), (expr13, temp > 1e4))

        LDL = cool
        
        factor = sp.Piecewise((LDL*HDL / (LDL + HDL), sp.And(LDL >= 1e-99, HDL >=1e-99)), (0, True))

        if self.massfracs == 1:
            expr14_old = factor * composition[ChemSpecie('h2').sym_name]*(density/ChemSpecie('h2').m) * \
                     sp.Min(1.0, sp.Max(1.25e-10 * self.get_n(composition, density), 1e-40)**(-0.45)) #erg/cm3/s

        elif self.massfracs == 0:
            expr14_old = factor * composition[ChemSpecie('h2').sym_name] * \
                     sp.Min(1.0, sp.Max(1.25e-10 * self.get_n(composition, density), 1e-40)**(-0.45)) #erg/cm3/s

        expr14 = expr14_old.rewrite(sp.Piecewise)

        CH2 = sp.Piecewise((0, temp < 2e0), (expr14, True))

        return CH2

    def get_compsubs(self, T, composition, density=0):

        if self.withD == 1:
            compsubs = [composition[self.get_allspecies_sym()[0]], composition[self.get_allspecies_sym()[1]], composition[self.get_allspecies_sym()[2]], \
                        composition[self.get_allspecies_sym()[3]], composition[self.get_allspecies_sym()[4]], composition[self.get_allspecies_sym()[5]], \
                        composition[self.get_allspecies_sym()[6]], composition[self.get_allspecies_sym()[7]], composition[self.get_allspecies_sym()[8]], \
                        composition[self.get_allspecies_sym()[9]], composition[self.get_allspecies_sym()[10]], composition[self.get_allspecies_sym()[11]], \
                        composition[self.get_allspecies_sym()[12]], composition[self.get_allspecies_sym()[13]], T]
            if self.massfracs == 1:
                compsubs.insert(0, density)

        elif self.withD == 0:
            compsubs = [composition[self.get_allspecies_sym()[0]], composition[self.get_allspecies_sym()[1]], composition[self.get_allspecies_sym()[2]], \
                        composition[self.get_allspecies_sym()[3]], composition[self.get_allspecies_sym()[4]], composition[self.get_allspecies_sym()[5]], \
                        composition[self.get_allspecies_sym()[6]], composition[self.get_allspecies_sym()[7]], composition[self.get_allspecies_sym()[8]], \
                        T]
            if self.massfracs == 1:
                compsubs.insert(0, density)

        return compsubs

    def get_lambdifys_ydots(self, redshift):
        import sympy as sp
        T = sp.symbols('T', positive=True)

        if self.massfracs == 1:
            density = sp.symbols('rho', positive=True)
        elif self.massfracs == 0:
            density = 0

        composition = ChemComposition(specie=self.get_allspecies()).sympy()

        ydots_elec = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('elec').sym_name], 'numpy')
        ydots_hp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('hp').sym_name], 'numpy')
        ydots_h = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('h').sym_name], 'numpy')
        ydots_hm = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('hm').sym_name], 'numpy')
        ydots_h2p = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('h2p').sym_name], 'numpy')
        ydots_h2 = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('h2').sym_name], 'numpy')
        ydots_hepp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('hepp').sym_name], 'numpy')
        ydots_hep = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('hep').sym_name], 'numpy')
        ydots_he = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('he').sym_name], 'numpy')
        #ydot_T should be found using math mode and not numpy, because of issues in 1/HDL, 1/LDL with sympy in H2 cooling
        ydots_T = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_tdot(T, composition, redshift, density), 'math')

        if self.withD == 1:
            ydots_dp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('dp').sym_name], 'numpy')
            ydots_d = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('d').sym_name], 'numpy')
            ydots_dm = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('dm').sym_name], 'numpy')
            ydots_hdp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('hdp').sym_name], 'numpy')
            ydots_hd = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_ydots(T, composition, density)[ChemSpecie('hd').sym_name], 'numpy')

            return [ydots_elec, ydots_hp, ydots_h, ydots_hm, ydots_dp, ydots_d, ydots_h2p, ydots_dm, ydots_h2, ydots_hdp, ydots_hd, ydots_hepp, \
                    ydots_hep, ydots_he, ydots_T]

        elif self.withD == 0:

            return [ydots_elec, ydots_hp, ydots_h, ydots_hm, ydots_h2p, ydots_h2, ydots_hepp, ydots_hep, ydots_he, ydots_T]


    def get_sumAbarinv(self,composition):
        summ = 0

        if self.massfracs == 1:
            for specie in composition:
                summ += composition[specie]/(self.sympy_to_python(specie).m/cons.p_mass)

        elif self.massfracs == 0:
            density = self.get_rho(composition)

            for specie in composition:
                summ += composition[specie]

            summ *= cons.p_mass/density

        return summ

    def get_sumgammasinv(self,composition):
        summ = 0

        if self.massfracs == 1:
            for specie in composition:
                summ += (composition[specie]/(self.sympy_to_python(specie).m/cons.p_mass)) * (1.0 / (self.sympy_to_python(specie).gamma-1))
            summ /= self.get_sumAbarinv(composition)

        elif self.massfracs == 0:
            density = self.get_rho(composition)

            for specie in composition:
                summ += (composition[specie]*cons.p_mass/density) * (1.0 / (self.sympy_to_python(specie).gamma-1))
            summ /= self.get_sumAbarinv(composition)

        return summ

    def convert_T_to_E(self,T,composition):
        eint = T * cons.Rgas_cgs * self.get_sumAbarinv(composition) * self.get_sumgammasinv(composition)
        return eint

    def evaluate_jacobian_T(self, T, composition, redshift, density=0):
        #for jacobians with respect to species and temperature
        import sympy as sp
        ydots = self.evaluate_ydots(T, composition, density)
        tdot = self.evaluate_tdot(T, composition, redshift, density)

        jacobian_elec = [sp.diff(ydots[ChemSpecie('elec').sym_name], specie) for specie in composition]
        jacobian_hp = [sp.diff(ydots[ChemSpecie('hp').sym_name], specie) for specie in composition]
        jacobian_h = [sp.diff(ydots[ChemSpecie('h').sym_name], specie) for specie in composition]
        jacobian_hm = [sp.diff(ydots[ChemSpecie('hm').sym_name], specie) for specie in composition]
        jacobian_h2p = [sp.diff(ydots[ChemSpecie('h2p').sym_name], specie) for specie in composition]
        jacobian_h2 = [sp.diff(ydots[ChemSpecie('h2').sym_name], specie) for specie in composition]
        jacobian_hepp = [sp.diff(ydots[ChemSpecie('hepp').sym_name], specie) for specie in composition]
        jacobian_hep = [sp.diff(ydots[ChemSpecie('hep').sym_name], specie) for specie in composition]
        jacobian_he = [sp.diff(ydots[ChemSpecie('he').sym_name], specie) for specie in composition]

        jacobian_elec.append(sp.diff(ydots[ChemSpecie('elec').sym_name], T))
        jacobian_hp.append(sp.diff(ydots[ChemSpecie('hp').sym_name], T))
        jacobian_h.append(sp.diff(ydots[ChemSpecie('h').sym_name], T))
        jacobian_hm.append(sp.diff(ydots[ChemSpecie('hm').sym_name], T))
        jacobian_h2p.append(sp.diff(ydots[ChemSpecie('h2p').sym_name], T))
        jacobian_h2.append(sp.diff(ydots[ChemSpecie('h2').sym_name], T))
        jacobian_hepp.append(sp.diff(ydots[ChemSpecie('hepp').sym_name], T))
        jacobian_hep.append(sp.diff(ydots[ChemSpecie('hep').sym_name], T))
        jacobian_he.append(sp.diff(ydots[ChemSpecie('he').sym_name], T))

        jacobian_T = [sp.diff(tdot, specie) for specie in composition]
        jacobian_T.append(sp.diff(tdot, T))

        if self.withD == 1:
            jacobian_dp = [sp.diff(ydots[ChemSpecie('dp').sym_name], specie) for specie in composition]
            jacobian_d = [sp.diff(ydots[ChemSpecie('d').sym_name], specie) for specie in composition]
            jacobian_dm = [sp.diff(ydots[ChemSpecie('dm').sym_name], specie) for specie in composition]
            jacobian_hdp = [sp.diff(ydots[ChemSpecie('hdp').sym_name], specie) for specie in composition]
            jacobian_hd = [sp.diff(ydots[ChemSpecie('hd').sym_name], specie) for specie in composition]

            jacobian_dp.append(sp.diff(ydots[ChemSpecie('dp').sym_name], T))
            jacobian_d.append(sp.diff(ydots[ChemSpecie('d').sym_name], T))
            jacobian_dm.append(sp.diff(ydots[ChemSpecie('dm').sym_name], T))
            jacobian_hdp.append(sp.diff(ydots[ChemSpecie('hdp').sym_name], T))
            jacobian_hd.append(sp.diff(ydots[ChemSpecie('hd').sym_name], T))

            return [jacobian_elec, jacobian_hp, jacobian_h, jacobian_hm, jacobian_dp, jacobian_d, jacobian_h2p, jacobian_dm, \
                    jacobian_h2, jacobian_hdp, jacobian_hd, jacobian_hepp, jacobian_hep, jacobian_he, jacobian_T]

        elif self.withD == 0:

            return [jacobian_elec, jacobian_hp, jacobian_h, jacobian_hm, jacobian_h2p, jacobian_h2, jacobian_hepp, jacobian_hep, \
                    jacobian_he, jacobian_T]


    def evaluate_jacobian_Eint(self, T, composition, redshift, density=0):
        #for jacobians with respect to species and internal energy (needed for AMREeX Microphysics)
        import sympy as sp
        ydots = self.evaluate_ydots(T, composition, density)
        tdot = self.evaluate_tdot(T, composition, redshift, density)
        Eint = self.convert_T_to_E(T, composition)

        jacobian_elec = [sp.diff(ydots[ChemSpecie('elec').sym_name], specie) for specie in composition]
        jacobian_hp = [sp.diff(ydots[ChemSpecie('hp').sym_name], specie) for specie in composition]
        jacobian_h = [sp.diff(ydots[ChemSpecie('h').sym_name], specie) for specie in composition]
        jacobian_hm = [sp.diff(ydots[ChemSpecie('hm').sym_name], specie) for specie in composition]
        jacobian_h2p = [sp.diff(ydots[ChemSpecie('h2p').sym_name], specie) for specie in composition]
        jacobian_h2 = [sp.diff(ydots[ChemSpecie('h2').sym_name], specie) for specie in composition]
        jacobian_hepp = [sp.diff(ydots[ChemSpecie('hepp').sym_name], specie) for specie in composition]
        jacobian_hep = [sp.diff(ydots[ChemSpecie('hep').sym_name], specie) for specie in composition]
        jacobian_he = [sp.diff(ydots[ChemSpecie('he').sym_name], specie) for specie in composition]

        #to get dY/dEint, we divide dY/dT by dEint/dT:
        dEint_dT = sp.diff(Eint, T)
        jacobian_elec.append(sp.diff(ydots[ChemSpecie('elec').sym_name], T) / dEint_dT)
        jacobian_hp.append(sp.diff(ydots[ChemSpecie('hp').sym_name], T) / dEint_dT)
        jacobian_h.append(sp.diff(ydots[ChemSpecie('h').sym_name], T) / dEint_dT)
        jacobian_hm.append(sp.diff(ydots[ChemSpecie('hm').sym_name], T) / dEint_dT)
        jacobian_h2p.append(sp.diff(ydots[ChemSpecie('h2p').sym_name], T) / dEint_dT)
        jacobian_h2.append(sp.diff(ydots[ChemSpecie('h2').sym_name], T) / dEint_dT)
        jacobian_hepp.append(sp.diff(ydots[ChemSpecie('hepp').sym_name], T) / dEint_dT)
        jacobian_hep.append(sp.diff(ydots[ChemSpecie('hep').sym_name], T) / dEint_dT)
        jacobian_he.append(sp.diff(ydots[ChemSpecie('he').sym_name], T) / dEint_dT)

        jacobian_T = [sp.diff(tdot, specie) for specie in composition]
        jacobian_T.append(sp.diff(tdot, T) / dEint_dT)

        if self.withD == 1:
            jacobian_dp = [sp.diff(ydots[ChemSpecie('dp').sym_name], specie) for specie in composition]
            jacobian_d = [sp.diff(ydots[ChemSpecie('d').sym_name], specie) for specie in composition]
            jacobian_dm = [sp.diff(ydots[ChemSpecie('dm').sym_name], specie) for specie in composition]
            jacobian_hdp = [sp.diff(ydots[ChemSpecie('hdp').sym_name], specie) for specie in composition]
            jacobian_hd = [sp.diff(ydots[ChemSpecie('hd').sym_name], specie) for specie in composition]

            jacobian_dp.append(sp.diff(ydots[ChemSpecie('dp').sym_name], T) / dEint_dT)
            jacobian_d.append(sp.diff(ydots[ChemSpecie('d').sym_name], T) / dEint_dT)
            jacobian_dm.append(sp.diff(ydots[ChemSpecie('dm').sym_name], T) / dEint_dT)
            jacobian_hdp.append(sp.diff(ydots[ChemSpecie('hdp').sym_name], T) / dEint_dT)
            jacobian_hd.append(sp.diff(ydots[ChemSpecie('hd').sym_name], T) / dEint_dT)


            return [jacobian_elec, jacobian_hp, jacobian_h, jacobian_hm, jacobian_dp, jacobian_d, jacobian_h2p, jacobian_dm, \
                    jacobian_h2, jacobian_hdp, jacobian_hd, jacobian_hepp, jacobian_hep, jacobian_he, jacobian_T]

        elif self.withD == 0:

            return [jacobian_elec, jacobian_hp, jacobian_h, jacobian_hm, jacobian_h2p, \
                    jacobian_h2, jacobian_hepp, jacobian_hep, jacobian_he, jacobian_T]


    def get_lambdifys_jacobian_T(self, redshift):
        import sympy as sp
        T = sp.symbols('T', positive=True)
        composition = ChemComposition(specie=self.get_allspecies()).sympy()
        if self.massfracs == 0:
            density = 0
        elif self.massfracs == 1:
            density = sp.symbols('rho', positive=True)

        if self.withD == 1:

            jac_elec = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[0], 'numpy')
            jac_hp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[1], 'numpy')
            jac_h = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[2], 'numpy')
            jac_hm = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[3], 'numpy')
            jac_dp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[4], 'numpy')
            jac_d = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[5], 'numpy')
            jac_h2p = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[6], 'numpy')
            jac_dm = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[7], 'numpy')
            jac_h2 = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[8], 'numpy')
            jac_hdp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[9], 'numpy')
            jac_hd = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[10], 'numpy')
            jac_hepp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[11], 'numpy')
            jac_hep = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[12], 'numpy')
            jac_he = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[13], 'numpy')
            jac_T = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[14], 'math')

            return [jac_elec, jac_hp, jac_h, jac_hm, jac_dp, jac_d, jac_h2p, jac_dm, jac_h2, jac_hdp, \
                    jac_hd, jac_hepp, jac_hep, jac_he, jac_T]

        elif self.withD == 0:

            jac_elec = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[0], 'numpy')
            jac_hp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[1], 'numpy')
            jac_h = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[2], 'numpy')
            jac_hm = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[3], 'numpy')
            jac_h2p = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[4], 'numpy')
            jac_h2 = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[5], 'numpy')
            jac_hepp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[6], 'numpy')
            jac_hep = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[7], 'numpy')
            jac_he = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[8], 'numpy')
            jac_T = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_T(T, composition, redshift, density)[9], 'math')

            return [jac_elec, jac_hp, jac_h, jac_hm, jac_h2p, jac_h2, jac_hepp, jac_hep, jac_he, jac_T]


    def get_lambdifys_jacobian_Eint(self, redshift):
        import sympy as sp
        T = sp.symbols('T', positive=True)
        composition = ChemComposition(specie=self.get_allspecies()).sympy()
        if self.massfracs == 0:
            density = 0
        elif self.massfracs == 1:
            density = sp.symbols('rho', positive=True)

        if self.withD == 1:

            jac_elec = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[0], 'numpy')
            jac_hp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[1], 'numpy')
            jac_h = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[2], 'numpy')
            jac_hm = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[3], 'numpy')
            jac_dp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[4], 'numpy')
            jac_d = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[5], 'numpy')
            jac_h2p = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[6], 'numpy')
            jac_dm = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[7], 'numpy')
            jac_h2 = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[8], 'numpy')
            jac_hdp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[9], 'numpy')
            jac_hd = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[10], 'numpy')
            jac_hepp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[11], 'numpy')
            jac_hep = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[12], 'numpy')
            jac_he = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[13], 'numpy')
            jac_T = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[14], 'math')

            return [jac_elec, jac_hp, jac_h, jac_hm, jac_dp, jac_d, jac_h2p, jac_dm, jac_h2, jac_hdp, \
                    jac_hd, jac_hepp, jac_hep, jac_he, jac_T]

        elif self.withD == 0:

            jac_elec = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[0], 'numpy')
            jac_hp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[1], 'numpy')
            jac_h = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[2], 'numpy')
            jac_hm = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[3], 'numpy')
            jac_h2p = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[4], 'numpy')
            jac_h2 = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[5], 'numpy')
            jac_hepp = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[6], 'numpy')
            jac_hep = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[7], 'numpy')
            jac_he = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[8], 'numpy')
            jac_T = sp.lambdify(self.get_compsubs(T, composition, density), self.evaluate_jacobian_Eint(T, composition, redshift, density)[9], 'math')

            return [jac_elec, jac_hp, jac_h, jac_hm, jac_h2p, jac_h2, jac_hepp, jac_hep, jac_he, jac_T]


    def fft_numdens(self, y):
        #when using number densities
        allspecies = self.get_allspecies()
        comp = ChemComposition(specie=(allspecies)).set_specie_numberdens(nval=(y))

        rho = comp[ChemSpecie('hp')]*ChemSpecie('hp').m + comp[ChemSpecie('h')]*ChemSpecie('h').m + \
              comp[ChemSpecie('hm')]*ChemSpecie('hm').m + comp[ChemSpecie('h2')]*ChemSpecie('h2').m + \
              comp[ChemSpecie('h2p')]*ChemSpecie('h2p').m + comp[ChemSpecie('he')]*ChemSpecie('he').m + \
              comp[ChemSpecie('hep')]*ChemSpecie('hep').m + comp[ChemSpecie('hepp')]*ChemSpecie('hepp').m + \
              comp[ChemSpecie('elec')]*ChemSpecie('elec').m             

        if self.withD == 1:
            rho += comp[ChemSpecie('hd')]*ChemSpecie('hd').m + comp[ChemSpecie('hdp')]*ChemSpecie('hdp').m + \
                   comp[ChemSpecie('d')]*ChemSpecie('d').m + comp[ChemSpecie('dm')]*ChemSpecie('dm').m + \
                   comp[ChemSpecie('dp')]*ChemSpecie('dp').m

        free_fall_time = np.sqrt(3.0*np.pi/32.0/cons.gravity/rho)
        return free_fall_time

    def rhs_numdens(self, t, y):
        #when using number densities
        ydots = self.ydots_lambdified

        if len(y) == 15: #self.withD ==1
            new_ydots = [ydots[x](y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], \
                                  y[10], y[11], y[12], y[13], y[14]) for x in np.arange(len(y))]
        elif len(y) == 10: #self.withD ==0
            new_ydots = [ydots[x](y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]) for x in np.arange(len(y))]

        return new_ydots

    def jac_numdens(self, t, y):
        #when using number densities
        
        jacs = self.jacs_lambdified

        if len(y) == 15:
            new_jacs = [jacs[x](y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], \
                                  y[10], y[11], y[12], y[13], y[14]) for x in np.arange(len(y))]
        elif len(y) == 10:
            new_jacs = [jacs[x](y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]) for x in np.arange(len(y))]

        return new_jacs


    def rhs_massfracs(self, t, y, density):
        #when using mass fractions
        ydots = self.ydots_lambdified

        if len(y) == 15:
            new_ydots = [ydots[x](density, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], \
                                  y[10], y[11], y[12], y[13], y[14]) for x in np.arange(len(y))]
        elif len(y) == 10:
            new_ydots = [ydots[x](density, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]) for x in np.arange(len(y))]

        return new_ydots

    def jac_massfracs(self, t, y, density):
        #when using mass fractions
        jacs = self.jacs_lambdified

        if len(y) == 15:
            new_jacs = [jacs[x](density, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], \
                                  y[10], y[11], y[12], y[13], y[14]) for x in np.arange(len(y))]
        elif len(y) == 10:
            new_jacs = [jacs[x](density, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]) for x in np.arange(len(y))]

        return new_jacs


    def write_cxxnetwork(self, T, composition, redshift, density=0):
        import sympy as sp
        from sympy.printing import cxxcode


        sy = []
        for i in range(0,len(composition)):
            sy.append(sp.symbols('X('+str(i)+')'))

        ydots = self.evaluate_ydots(T, composition, density)


        f = open('/scratch/jh2/ps3459/pynucastro/pynucastro/networks/kromechem_rhs_empty.H', 'r')
        fcontent = f.read()

        print('Expanding log')
        
        fupdated = ''
        for i, specie in enumerate(composition):
            ydots_subs = ydots[specie].subs({ChemSpecie('elec').sym_name: sy[0], ChemSpecie('hp').sym_name: sy[1], \
                                             ChemSpecie('h').sym_name: sy[2], ChemSpecie('hm').sym_name: sy[3], \
                                             ChemSpecie('dp').sym_name: sy[4], ChemSpecie('d').sym_name: sy[5], \
                                             ChemSpecie('h2p').sym_name: sy[6], ChemSpecie('dm').sym_name: sy[7], \
                                             ChemSpecie('h2').sym_name: sy[8], ChemSpecie('hdp').sym_name: sy[9], \
                                             ChemSpecie('hd').sym_name: sy[10], ChemSpecie('hepp').sym_name: sy[11], \
                                             ChemSpecie('hep').sym_name: sy[12], ChemSpecie('he').sym_name: sy[13]})
            fupdated += '    ydot(' + str(i+1) + ') = ' + cxxcode(sp.expand_log(ydots_subs)) + ';\n\n\n'
            print('Expanded log of ydot ', i+1)


        fcontent = fcontent.replace('    <ydot>', fupdated)
        print('YDOTS written')

        #put tdot in place of edot in the C++ script
        tdot = self.evaluate_tdot(T, composition, redshift, density)
        tdot_subs = tdot.subs({ChemSpecie('elec').sym_name: sy[0], ChemSpecie('hp').sym_name: sy[1], \
                             ChemSpecie('h').sym_name: sy[2], ChemSpecie('hm').sym_name: sy[3], \
                             ChemSpecie('dp').sym_name: sy[4], ChemSpecie('d').sym_name: sy[5], \
                             ChemSpecie('h2p').sym_name: sy[6], ChemSpecie('dm').sym_name: sy[7], \
                             ChemSpecie('h2').sym_name: sy[8], ChemSpecie('hdp').sym_name: sy[9], \
                             ChemSpecie('hd').sym_name: sy[10], ChemSpecie('hepp').sym_name: sy[11], \
                             ChemSpecie('hep').sym_name: sy[12], ChemSpecie('he').sym_name: sy[13]})
        edotstr = cxxcode(sp.expand_log(tdot_subs))
        fcontent = fcontent.replace('<edot>', edotstr)
        print('EDOT written')


        jacs = self.evaluate_jacobian_Eint(T, composition, redshift, density)

        fupdated = ''
        for i in range(len(composition)+1):
            for j in range(len(composition)+1):
                #f.write('/* JAC ' + str(splist[i]) + str(splist[j]) + ' */')
                #f.write('\n')
                jacs_subs = jacs[i][j].subs({ChemSpecie('elec').sym_name: sy[0], ChemSpecie('hp').sym_name: sy[1], \
                                                 ChemSpecie('h').sym_name: sy[2], ChemSpecie('hm').sym_name: sy[3], \
                                                 ChemSpecie('dp').sym_name: sy[4], ChemSpecie('d').sym_name: sy[5], \
                                                 ChemSpecie('h2p').sym_name: sy[6], ChemSpecie('dm').sym_name: sy[7], \
                                                 ChemSpecie('h2').sym_name: sy[8], ChemSpecie('hdp').sym_name: sy[9], \
                                                 ChemSpecie('hd').sym_name: sy[10], ChemSpecie('hepp').sym_name: sy[11], \
                                                 ChemSpecie('hep').sym_name: sy[12], ChemSpecie('he').sym_name: sy[13]})
                fupdated += '    jac(' + str(i+1) + ',' + str(j+1) +') = ' + cxxcode(sp.expand_log(jacs_subs)) + ';\n\n\n'
                print('Expanded log of jac ', i+1, j+1)


        fcontent = fcontent.replace('    <jac>', fupdated)
        print('JACS written')


        g = open('/scratch/jh2/ps3459/pynucastro/pynucastro/networks/actual_rhs.H', 'w')
        g.write(fcontent)
        g.close()


        #f.close()
        return None

