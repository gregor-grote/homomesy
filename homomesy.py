from ipywidgets import interact_manual, interact, fixed
import ipywidgets as widgets
from sage.all import *
from sage.databases.findstat import FindStatMaps, FindStatStatistics
import requests
import json
findstat()._allow_execution = True
from sage.rings.rational_field import QQ
from dataclasses import dataclass
from sage.dynamics.finite_dynamical_system import FiniteDynamicalSystem
from sage.structure.parent import Parent
import matplotlib.pyplot as plt
import numpy as np
from itertools import product



findstat_collection = (
    (Permutations, "Permutations", Permutation),
    (ParkingFunctions, "Parking functions", ParkingFunction),
    # (Partitions, "Integer partitions", Partition),
    # (PlanePartitions, "Plane partitions", PlanePartition),
    # (DyckWords, "Dyck paths", DyckWord),
)


@dataclass
class CollectionWithMapsAndStats:
    collection: object
    name: str
    stats: list
    maps: list
    element_constructor: callable


all_collections: dict[str, CollectionWithMapsAndStats] = {}

for c, n, e in findstat_collection:
    print(f"Downloading data for {n}")
    maps = FindStatMaps(domain=n, codomain=n)
    maps = list(filter(lambda x: x.properties_raw().find("bijective") >= 0, maps))
    collection = CollectionWithMapsAndStats(c, n, list(FindStatStatistics(n)), maps, e)
    all_collections[n] = collection
print("done downloading data")
print("printing stats...")
print(all_collections)

class FunctionWithName:
    def __init__(self, f, name):
        self.f = f
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class ListWrapper:
    def __init__(self, lst):
        if isinstance(lst, self.__class__):
            self.lst = list(lst.lst)
            self.n = lst.n
        else:
            self.lst = list(lst)
            self.n = len(lst)

    def __iter__(self):
        return iter(self.lst)

    def __getitem__(self, i):
        return self.lst[i]

    def __len__(self):
        return len(self.lst)

    def __str__(self):
        return str(self.lst)

    def __repr__(self):
        return str(self.lst)
    
    def __eq__(self, value):
        try:
            return self.lst == list(value)
        except:
            return False
        
    def __hash__(self):
        return hash(tuple(self.lst))
    
class IntegerListWrapper(ListWrapper):
    def number_of_fixed_points(self):
        return sum([1 for i in range(len(self)) if i + 1 == self[i]])
    
    
def is_fubini_ranking(lst: list):
    n = len(lst)
    for i in range(1, n + 1):
        count = lst.count(i)
        for j in range(1, count):
            if i + j in lst:
                return False
    return True


def generate_fubini_rankings(n: int):
    p = ParkingFunctions(n)
    for pf in p:
        if is_fubini_ranking(pf):
            yield pf

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.n == other.n and self.lst == other.lst

    def __hash__(self):
        return hash(tuple(self.lst))


class FubiniRanking(IntegerListWrapper):
    def reverse(self):
        return FubiniRanking(list(reversed(self.lst)))


class FubiniRankings(Parent, ListWrapper):
    def __init__(self, n):
        if isinstance(n, list):
            ListWrapper.__init__(self, list([FubiniRanking(x) for x in n]))
        elif isinstance(n, int):
            ListWrapper.__init__(
                self, list([FubiniRanking(x) for x in generate_fubini_rankings(n)])
            )
        else:
            raise ValueError("n must be either a list or an integer")

    def __str__(self):
        return f"Fubini ranking of size {self.n}"

    def __repr__(self):
        return f"Fubini ranking of size {self.n}"


all_collections["Fubini rankings"] = CollectionWithMapsAndStats(
    FubiniRankings, "Fubini rankings", [], [], FubiniRanking
)
list(FubiniRankings(3))

# yoinked from https://11011110.github.io/blog/2013/03/13/cayley-permutations.html
def generate_cayley_permutations(n):
    """Generate sequence of Cayley permutations of length n"""
    if n < 2:
        yield [1] * n
        return
    for P in generate_cayley_permutations(n - 1):
        m = max(P)
        i = n - 1
        P = P + [m + 1]
        pastMax = False
        yield P
        while i > 0:
            if not pastMax:
                P[i] = m
                yield P
                if P[i - 1] == m:
                    pastMax = True
            P[i] = P[i - 1]
            P[i - 1] = m + 1
            i -= 1
            yield P


class CayleyPermutation(IntegerListWrapper):
    pass


class CayleyPermutations(Parent, ListWrapper):
    def __init__(self, n):
        if isinstance(n, list):
            ListWrapper.__init__(self, list([CayleyPermutation(x) for x in n]))
        elif isinstance(n, int):
            ListWrapper.__init__(
                self,
                list([CayleyPermutation(x) for x in generate_cayley_permutations(n)]),
            )
        else:
            raise ValueError("n must be either a list or an integer")

    def __str__(self):
        return f"Cayley permutation of size {self.n}"

    def __repr__(self):
        return f"Cayley permutation of size {self.n}"


all_collections["Cayley permutations"] = CollectionWithMapsAndStats(
    CayleyPermutations, "Cayley permutations", [], [], CayleyPermutation
)
list(CayleyPermutations(3))

def generate_tuples(n, iter_to=None):
    if iter_to is None:
        iter_to = n
    if n == 0:
        yield []
        return
    for p in generate_tuples(n - 1, iter_to):
        for i in range(1, iter_to + 1):
            yield [i] + p


class NTuple(IntegerListWrapper):
    pass


class NTuples(Parent, ListWrapper):
    def __init__(self, n):
        if isinstance(n, list):
            ListWrapper.__init__(self, list([NTuple(x) for x in n]))
        elif isinstance(n, int):
            ListWrapper.__init__(self, list([NTuple(x) for x in generate_tuples(n)]))
        else:
            raise ValueError("n must be either a list or an integer")

    def __str__(self):
        return f"All tuples of size {self.n}"

    def __repr__(self):
        return f"All tuples of size {self.n}"


def get_add_n_map(n):
    def add_n(p):
        return NTuple([(x + n - 1) % len(p) + 1 for x in p])

    return FunctionWithName(add_n, f"Add {n} to each element")


def get_subtract_n_map(n):
    def subtract_n(p):
        return NTuple([(x - n - 1) % len(p) + 1 for x in p])

    return FunctionWithName(subtract_n, f"Subtract {n} from each element")


def get_mult_n_map(n):
    def mult_n(p):
        return NTuple([(x * n - 1) % len(p) + 1 for x in p])

    return FunctionWithName(mult_n, f"Multiply each element by {n}")


fixed_point_statistic = FunctionWithName(IntegerListWrapper.number_of_fixed_points, "Number of fixed points")
cosine = FunctionWithName(lambda x: sum(x[i]*(i+1) for i in range(len(x))), "Cosine")

id = FunctionWithName(lambda x: x, "Identity")
reverse = FunctionWithName(lambda x: NTuple(list(reversed(list(x)))), "Reverse")
complement = FunctionWithName(lambda x: NTuple(list(len(x) - e + 1 for e in x)), "Complement")


all_collections["n-tuples"] = CollectionWithMapsAndStats(
    NTuples,
    "n-tuples",
    [cosine, fixed_point_statistic],
    [get_add_n_map(n) for n in range(1, 6)]
    #+ [get_subtract_n_map(n) for n in range(1, 6)]
    + [get_mult_n_map(n) for n in range(1, 6)]
    + [id, reverse, complement],
    NTuple,
)

list(NTuples(3))

def generate_inversion_sequences(n):
    for p in Permutations(n):
        yield [sum([1 for j in range(i) if p[j] > p[i]]) for i in range(n)]


class InversionSequence(IntegerListWrapper):
    pass


class InversionSequences(Parent, ListWrapper):
    def __init__(self, n):
        if isinstance(n, list):
            ListWrapper.__init__(self, list([InversionSequence(x) for x in n]))
        elif isinstance(n, int):
            ListWrapper.__init__(
                self,
                list([InversionSequence(x) for x in generate_inversion_sequences(n)]),
            )
        else:
            raise ValueError("n must be either a list or an integer")

    def __str__(self):
        return f"Inversion sequences of size {self.n}"

    def __repr__(self):
        return f"Inversion sequences of size {self.n}"


def get_add_k_map_zero_based(k):
    def add_k(p):
        return InversionSequence([(x + k) % len(p) for x in p])

    return FunctionWithName(add_k, f"Add {k} to each element zero-based")


def get_subtract_k_map_zero_based(k):
    def subtract_k(p):
        return InversionSequence([(x - k) % len(p) for x in p])
    
    return FunctionWithName(subtract_k, f"Subtract {k} from each element zero-based")


def get_mult_k_map_zero_based(k):
    def mult_k(p):
        return InversionSequence([(x * k) % len(p) for x in p])

    return FunctionWithName(mult_k, f"Multiply each element by {k} zero-based")


all_collections["Inversion sequences"] = CollectionWithMapsAndStats(
    InversionSequences,
    "Inversion sequences",
    [
        FunctionWithName(
            lambda l: len([1 for i in range(len(l)) if l[i] == i]),
            "Number of fixed points zero-based",
        ),
    ],
    [get_add_k_map_zero_based(k) for k in range(1, 6)]
    #+ [get_subtract_k_map_zero_based(k) for k in range(1, 6)]
    + [get_mult_k_map_zero_based(k) for k in range(1, 6)],
    InversionSequence,
)


def ana(name, **kwargs):
    def wrapper(f):
        f.__ana_name__ = name
        f.__ana_kwargs__ = kwargs
        return f

    return wrapper


class ListAnalysis(IntegerListWrapper):
    @ana("Number of each number")
    def number_counts(self):
        return ListAnalysis([sum([1 for x in self if x == i]) for i in range(1, len(self) + 1)])

    @ana("Number of unique elements")
    def unique_elements(self):
        return len(set(self))

    @ana(
        "Number of sequences of length {seq_len} that are increasing by {increase_by}",
        seq_len=(2, "n+1"),
        increase_by=(-1, 2),
    )
    def equally_increasing_sequences(self, seq_len, increase_by=0):
        r = [0] * (len(self))
        for i in range(len(self) - seq_len + 1):
            if all([self[i + j] == self[i] + j * increase_by and self[i] > 0 for j in range(seq_len)]):
                r[self[i] - 1] += 1
        return ListAnalysis(r)

    @ana(
        "Number of sequences of length {seq_len} that are increasing by {increase_by} with a loop",
        seq_len=(2, "n+1"),
        increase_by=(-1, 2),
    )
    def equally_increasing_sequences_with_loop(self, seq_len, increase_by=0):
        r = [0] * (len(self))
        for i in range(len(self)):
            if all(
                [
                    self[(i + j) % len(self)] == self[i] + j * increase_by and self[i] > 0
                    for j in range(seq_len)
                ]
            ):
                r[self[i] - 1] += 1
        return ListAnalysis(r)

    @ana("Exists number")
    def exists_number(self):
        return list([True if self[i] > 0 else False for i in range(len(self))])
    
    @ana("Element greater then zero")
    def element_greater_than_zero(self):
        return list([True if x > 0 else False for x in self])
    

    @ana("Number of elements less than each number")
    def elements_less_than(self):
        return ListAnalysis(
            [sum([1 for j in self if j < i]) for i in range(1, len(self) + 1)]
        )

    @ana("Number of elements greater than each number")
    def elements_greater_than(self):
        return ListAnalysis(
            [sum([1 for j in self if j > i]) for i in range(1, len(self) + 1)]
        )

    @ana("Number of elements less than each element to the left")
    def elements_to_the_left_smaller(self):
        return ListAnalysis(
            [sum([1 for j in self[:i - 1] if j < self[i - 1]]) for i in range(1, len(self) + 1)]
        )

    @ana("Number of elements less than each element to the right")
    def elements_to_the_right_smaller(self):
        return ListAnalysis(
            [sum([1 for j in self[i:] if j < self[i - 1]]) for i in range(1, len(self) + 1)]
        )

    @ana("Number of elements greater than each element to the left")
    def elements_to_the_left_larger(self):
        return ListAnalysis(
            [sum([1 for j in self[:i - 1] if j > self[i - 1]]) for i in range(1, len(self) + 1)]
        )

    @ana("Number of elements greater than each element to the right")
    def elements_to_the_right_larger(self):
        return ListAnalysis(
            [sum([1 for j in self[i:] if j > self[i - 1]]) for i in range(1, len(self) + 1)]
        )

    def get_all_analyser(self):
        return [
            getattr(self, x)
            for x in dir(self)
            if hasattr(getattr(self, x), "__ana_name__")
        ]

    def create_analysis(self, depth=2, prefix=tuple(), results=None, filter_zero_sequences=True):
        if results is None:
            results = {}
        if depth <= 0:
            return
        for a in self.get_all_analyser():
            if a.__ana_kwargs__:
                keys = list(a.__ana_kwargs__.keys())
                values = list(a.__ana_kwargs__.values())
                mapped_values = []
                for i in range(len(values)):
                    mapped_values.append([0,0]) 
                    for j in range(2):
                        if isinstance(values[i][j], str):
                            mapped_values[i][j] = values[i][j].replace("n", str(len(self)))
                            mapped_values[i][j] = int(eval(mapped_values[i][j]))
                        else:
                            mapped_values[i][j] = values[i][j]
                    mapped_values[i] = range(*mapped_values[i])
                for p in product(*mapped_values):
                    r = a(*p)
                    new_prefix = prefix + tuple([a.__ana_name__.format(**dict(zip(keys, p)))])
                    results[new_prefix] = r
                    if isinstance(r, ListAnalysis) and any(x > 0 for x in r):
                        r.create_analysis(depth=depth - 1, prefix=new_prefix, results=results)
            else:
                r = a()
                new_prefix = prefix + tuple([a.__ana_name__])
                results[new_prefix] = r
                if isinstance(r, ListAnalysis) and any(x > 0 for x in r):
                    r.create_analysis(depth=depth - 1, prefix=new_prefix, results=results)
        
        if len(prefix) == 0 and filter_zero_sequences:
            return dict(filter(lambda x: not isinstance(x[1], ListAnalysis) or not all([y == 0 for y in x[1]]), results.items()))
        return results


def find_patterns(elements, depth=2, filter_zero_sequences=False):
    elements = list([ListAnalysis(x) for x in elements])
    ana_results = list([x.create_analysis(depth=depth, filter_zero_sequences=filter_zero_sequences) for x in elements])
    keys = set(ana_results[0].keys())
    for r in ana_results:
        keys.intersection_update(r.keys())
    
    results = {}
    
    for key in keys:
        if all([r[key] == ana_results[0][key] for r in ana_results]):
            results[key] = ana_results[0][key]
            
    return results, ana_results

def format_row(x, y, ins):
    if ins:
        return f"{x}: {y}"
    else:
        return f"({x}: {y})"


def print_row(avg, cycle, stats, in_set):
    t = [format_row(x, y, inss) for x, y, inss in zip(cycle, stats, in_set)]
    if all(in_set):
        print(f"{avg}:\t {' -> '.join(t)}")
    else:
        print(f"*{avg}:\t {' -> '.join(t)}")


@dataclass
class HomomesyResult:
    all_avgs: list
    all_cycles: list
    all_stats: list
    all_in_sets: list

    def is_homomesic(self):
        return all([x == self.all_avgs[0] for x in self.all_avgs])

    def is_homomesic_in_set(self):
        return all(
            [
                x == self.all_avgs[0]
                for x, _ in filter(
                    lambda x: all(x[1]), zip(self.all_avgs, self.all_in_sets)
                )
            ]
        )

    def is_homometric(self):
        len_queue_queue = list(set([len(c) for c in self.all_cycles]))
        is_homometric = True
        for l in len_queue_queue:
            avgs = list(
                [a for a, x in zip(self.all_avgs, self.all_cycles) if len(x) == l]
            )
            if not all([x == avgs[0] for x in avgs]):
                is_homometric = False
                break
        return is_homometric

    def is_homometric_in_set(self):
        len_queue_queue = list(set([len(c) for c in self.all_cycles]))
        is_homometric = True
        for l in len_queue_queue:
            avgs = list(
                [a for a, x in zip(self.all_avgs, self.all_cycles) if len(x) == l]
            )
            in_sets = list(
                [x for x, y in zip(self.all_in_sets, self.all_cycles) if len(y) == l]
            )
            if not all([x == avgs[0] for x, y in zip(avgs, in_sets) if all(y)]):
                is_homometric = False
                break
        return is_homometric

    def get_grouped_orbits(self):
        avg_queue = list(set(self.all_avgs))
        grouped_orbits = {}
        while avg_queue:
            avg = avg_queue.pop()
            grouped_orbits[avg] = HomomesyResult([], [], [], [])
            for a, c, s, ins in zip(
                self.all_avgs, self.all_cycles, self.all_stats, self.all_in_sets
            ):
                if a == avg:
                    grouped_orbits[avg].all_avgs.append(a)
                    grouped_orbits[avg].all_cycles.append(c)
                    grouped_orbits[avg].all_stats.append(s)
                    grouped_orbits[avg].all_in_sets.append(ins)
        return grouped_orbits

    def get_all_elements(self):
        return sum(self.all_cycles, [])

    def find_patterns(self, depth=2, filter_zero_sequences=False):
        return find_patterns(
            self.get_all_elements(),
            depth=depth,
            filter_zero_sequences=filter_zero_sequences,
        )

    def print_all_cycles(self):
        for a, c, s, ins in zip(
            self.all_avgs, self.all_cycles, self.all_stats, self.all_in_sets
        ):
            print_row(a, c, s, ins)

    def print_result(self):
        print(f"Homomesic: {self.is_homomesic()}")
        print(f"Homomesic using only cycles in set: {self.is_homomesic_in_set()}")
        print(f"Homometric: {self.is_homometric()}")
        print(f"Homometric using only cycles in set: {self.is_homometric_in_set()}")

    def print_grouped_orbits(self):
        for avg, res in self.get_grouped_orbits().items():
            print(f"Orbit with average {avg}")
            print()
            res.print_all_cycles()
            print()

    def find_unique_patterns_per_group(self, depth=2, filter_zero_sequences=False):
        grouped_orbits = list(self.get_grouped_orbits().values())
        all_patterns_per_group, full_pattern_list = zip(
            *[
                x.find_patterns(
                    depth=depth, filter_zero_sequences=filter_zero_sequences
                )
                for x in grouped_orbits
            ]
        )

        unique_patterns = [{} for _ in range(len(all_patterns_per_group))]
        for i in range(len(all_patterns_per_group)):
            for pattern, val in all_patterns_per_group[i].items():
                unique = True
                for j in range(len(full_pattern_list)):
                    if i != j:
                        for other_pattern_dict in full_pattern_list[j]:
                            if other_pattern_dict.get(pattern, None) == val:
                                unique = False
                                break
                        if not unique:
                            break

                if unique:
                    unique_patterns[i][pattern] = val

        def filter_dict(d):
            return dict(filter(lambda x: not (x[0][:-1] in d), d.items()))

        return [filter_dict(x) for x in unique_patterns], grouped_orbits

    def print_grouped_orbits_with_unique_patterns(
        self, depth=2, filter_zero_sequences=False
    ):
        all_patterns, all_orbits = self.find_unique_patterns_per_group(
            depth=depth, filter_zero_sequences=filter_zero_sequences
        )

        for patterns, orbits in zip(all_patterns, all_orbits):
            print(f"Orbits with average {orbits.all_avgs[0]}\n")
            if len(orbits.get_all_elements()) > 1:
                for pattern, val in patterns.items():
                    print(f"Unique pattern found: {' -> '.join(pattern)}: {val}")
            print()
            orbits.print_all_cycles()
            print("---------\n")

    def print_latex_orbit_code(
        self,
        main_circle_radius=6,
        orbit_circle_radius_map={1: 0, 2: 1, 4: 2, 8: 3},
        print_function=lambda p: "$(" + ", ".join([str(e) for e in p]) + ")$",
        x_stretch=1.2,
        print_avg=True,
    ):
        main_step_size = 2 * np.pi / len(self.all_cycles)
        for i, O in enumerate(self.all_cycles):
            x_center = main_circle_radius * np.sin(i * main_step_size)
            x_center *= x_stretch
            y_center = main_circle_radius * np.cos(i * main_step_size)
            orbit_step_size = 2 * np.pi / len(O)
            for j, p in enumerate(O):
                x = x_center + orbit_circle_radius_map[len(O)] * np.sin(
                    j * orbit_step_size
                )
                x *= x_stretch
                y = y_center + orbit_circle_radius_map[len(O)] * np.cos(
                    j * orbit_step_size
                )
                print(
                    f"\\node ({chr(i+97)}{j}) at ({x:.3f}, {y:.3f}) {{{print_function(p)}}};"
                )
            print()

            def get_anchor_start(j):
                name = chr(i + 97) + str(j)
                if j == 0:
                    return name + ".east"
                elif j < len(O) / 2:
                    return name + ".south"
                elif j == len(O) / 2:
                    return name + ".west"
                else:
                    return name + ".north"

            def get_anchor_end(j):
                name = chr(i + 97) + str(j % len(O))
                if j % len(O) == 0:
                    return name + ".west"
                elif j < len(O) / 2:
                    return name + ".north"
                elif j == len(O) / 2:
                    return name + ".east"
                else:
                    return name + ".south"

            def get_start_angle(j):
                angle = int(j * orbit_step_size / np.pi * 180)
                return (-angle + 720) % 360

            def get_end_angle(j):
                angle = int(j * orbit_step_size / np.pi * 180)
                return (-angle + 180 + 720) % 360

            for j in range(len(O)):
                print(
                    f"\\draw[->] ({get_anchor_start(j)}) to[out={get_start_angle(j)}, in={get_end_angle(j + 1)}] ({get_anchor_end(j + 1)});"
                )
            if print_avg:
                print()
                print(f"\\node at ({x_center*x_stretch:.3f}, {y_center:.3f}) {{{self.all_avgs[i]}}};")
            print("\n")
            
def lazy_cycles(F):
    visited = set()
    for e in F:
        if e in visited:
            continue
        orb = [e]
        in_set = [True]
        visited.add(e)
        curr = F._phi(e)
        while curr != e:
            if curr in orb:
                raise ValueError("Function is not a bijection. Orbit: " + " -> ".join([str(x) for x in orb + [curr]]))
            in_set.append(curr in F)
            visited.add(curr)
            orb.append(curr)
            curr = F._phi(curr)
        yield orb, in_set

def is_homomesic_with_debug(F, stat, pre_stat_map=lambda x: x, ignore_errors=False):
    all_avgs = []
    all_cycles = []
    all_stats = []
    all_in_sets = []


    for cyc, in_set in lazy_cycles(F):
        try:
            l = len(cyc)
            cur_cycle = list(cyc)
            cur_stats = [stat(pre_stat_map(x)) for x in cur_cycle]
            cur_avg = ~(QQ(l)) * sum(cur_stats)
            all_avgs.append(cur_avg)
            all_cycles.append(cur_cycle)
            all_stats.append(cur_stats)
            all_in_sets.append(in_set)
        except Exception as e:
            if ignore_errors:
                continue
            else:
                raise e


    return HomomesyResult(all_avgs, all_cycles, all_stats, all_in_sets)

def is_homomesic_with_debug_and_set_cycles(cycles, stat, pre_stat_map=None, ignore_errors=False):
    all_avgs = []
    all_stats = []
    all_in_sets = []
    all_cycles = [cyc for cyc, _ in cycles]
    
    for cyc, in_set in cycles:
        try:
            l = len(cyc)
            cur_cycle = list(cyc)
            if pre_stat_map is None:
                cur_stats = [stat(x) for x in cur_cycle]
            else:
                cur_stats = [stat(pre_stat_map(x)) for x in cur_cycle]
            cur_avg = ~(QQ(l)) * sum(cur_stats)
            all_avgs.append(cur_avg)
            all_stats.append(cur_stats)
            all_in_sets.append(in_set)
        except Exception as e:
            if ignore_errors:
                continue
            else:
                raise e
            
    return HomomesyResult(all_avgs, all_cycles, all_stats, all_in_sets)
    
def first_entry(lst: IntegerListWrapper) -> int:
    return lst[0]

first_entry.__name__ = "Stat 55: First entry of the sequence"

def the_number_of_excedances(lst: IntegerListWrapper) -> int:
    return sum([1 for i in range(len(lst)) if lst[i] > i + 1])

the_number_of_excedances.__name__ = "Stat 155: Number of excedances"

def the_number_of_weak_excedances(lst: IntegerListWrapper) -> int:
    return sum([1 for i in range(len(lst)) if lst[i] >= i + 1])

the_number_of_weak_excedances.__name__ = "Stat 213: Number of weak excedances"

def the_number_of_non_cyclical_small_weak_excedances(lst: IntegerListWrapper) -> int:
    n = len(lst)
    return sum(1 for i in range(n) if lst[i] != ((i+1) % n) + 1)

the_number_of_non_cyclical_small_weak_excedances.__name__ = "Stat 235: Number of indices that are not cyclical small weak excedances"

def the_number_of_cyclical_small_weak_excedances(lst: IntegerListWrapper) -> int:
    n = len(lst)
    return sum( 1 for i in range(n) if lst[i] in [ i+1, (i+1) % n + 1 ] )

the_number_of_cyclical_small_weak_excedances.__name__ = "Stat 236: Number of indices that are cyclical small weak excedances"

def the_number_of_small_excedances(lst: IntegerListWrapper) -> int:
    return sum( 1 for i in range(len(lst)) if lst[i] == i+2 )

the_number_of_small_excedances.__name__ = "Stat 237: Number of small excedances"


def the_number_of_non_small_weak_excedances(lst: IntegerListWrapper) -> int:
    return sum(1 for i in range(len(lst)) if lst[i] != i+1 and lst[i] != i+2)

the_number_of_non_small_weak_excedances.__name__ = "Stat 238: Number of indices that are not small weak excedances"

def the_number_of_small_weak_excedances(lst: IntegerListWrapper) -> int:
    return sum( 1 for i in range(len(lst)) if lst[i] in [i+1,i+2]  )

the_number_of_small_weak_excedances.__name__ = "Stat 239: Number of small weak excedances"

def the_number_of_non_small_excedances(lst: IntegerListWrapper) -> int:
    return sum(1 for i in range(len(lst)) if lst[i] != i+2)

the_number_of_non_small_excedances.__name__ = "Stat 240: Number of indices that are not small excedances"

def the_number_of_cyclical_small_excedances(lst: IntegerListWrapper) -> int:
        n = len(lst)
        return sum( 1 for i in range(n) if lst[i] == ( (i+1) % n + 1 ) )

the_number_of_cyclical_small_excedances.__name__ = "Stat 241: Number of indices that are cyclical small weak excedances"

def the_number_of_non_cyclical_small_excedances(lst: IntegerListWrapper) -> int:
        n = len(lst)
        return sum( 1 for i in range(n) if lst[i] not in [ i+1, (i+1) % n + 1 ] )
    
the_number_of_non_cyclical_small_excedances.__name__ = "Stat 242: Number of indices that are not cyclical small weak excedances"

IntegerListWrapper.number_of_fixed_points.__name__ = "Stat 22: Number of fixed points"

general_statistics = [
    first_entry,
    IntegerListWrapper.number_of_fixed_points,
    the_number_of_excedances,
    the_number_of_weak_excedances,
    the_number_of_non_cyclical_small_weak_excedances,
    the_number_of_cyclical_small_weak_excedances,
    the_number_of_small_excedances,
    the_number_of_non_small_weak_excedances,
    the_number_of_small_weak_excedances,
    the_number_of_non_small_excedances,
    the_number_of_cyclical_small_excedances,
    the_number_of_non_cyclical_small_excedances,
]