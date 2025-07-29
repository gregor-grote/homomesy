from sage.all import (
    Words,
    Word,
    Permutation,
    Permutations,
    ParkingFunction,
    ParkingFunctions,
    findstat,
)
from sage.rings.rational_field import QQ
from sage.structure.unique_representation import UniqueRepresentation
from sage.combinat.combinat import CombinatorialElement
from sage.databases.findstat import FindStatMaps, FindStatStatistics

findstat()._allow_execution = True

from dataclasses import dataclass
from sage.structure.parent import Parent
import numpy as np
from itertools import product
from typing import Literal, Callable, Iterable


# Helper classes


class NamedFunction:
    def __init__(
        self,
        f: callable,
        name: str | None = None,
    ):
        self.f = f
        self.__name__ = name if name is not None else str(f)

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


class FindStatFunction(NamedFunction):
    def __init__(self, f, id=None, name=None, full_name=None):
        super().__init__(f, name=name)
        self._id = id if id is not None else f.id_str()
        if name is not None:
            self.__name__ = f"{id}: {name}"
        elif full_name is not None:
            self.__name__ = full_name

    @staticmethod
    def from_new_code(f, new_code):
        fsf = FindStatFunction(f)
        fsf.f = new_code
        return fsf

    def id(self):
        return self._id

    def __eq__(self, value):
        if isinstance(value, FindStatFunction):
            return self.id() == value.id()
        elif isinstance(value, str):
            return self.id() == value
        elif isinstance(value, Callable):
            return self.id() == value.__name__
        else:
            return False


@dataclass
class CollectionWithMapsAndStats:
    collection: Callable[[int], Iterable[Iterable[int]]]
    name: str
    stats: list[NamedFunction]
    maps: list[NamedFunction]
    element_constructor: Callable[[Iterable[int]], Iterable[int]]


class HashableList[T](Iterable[T]):
    def __init__(self, lst):
        if isinstance(lst, self.__class__):
            self.lst = list(lst.lst)
        else:
            self.lst = list(lst)

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

    def __add__(self, other):
        if isinstance(other, HashableList):
            return HashableList(self.lst + other.lst)
        elif isinstance(other, list):
            return HashableList(self.lst + other)
        else:
            raise TypeError(f"Cannot add {type(other)} to HashableList")

    def __eq__(self, value):
        try:
            return self.lst == list(value)
        except:
            return False

    def __hash__(self):
        return hash(tuple(self.lst))


class PreMapper:
    def __init__(self, f, pre):
        self.func = f
        self.pre = pre

    def __call__(self, *args, **kwds):
        return self.func(self.pre(*args, **kwds))


class PostMapper:
    def __init__(self, f, post):
        self.func = f
        self.post = post

    def __call__(self, *args, **kwds):
        return self.post(self.func(*args, **kwds))


# Helper functions


def does_map_project_outside_set(f, S):
    for e in S:
        if f(e) not in S:
            return True
    return False


# Getting data

all_collections: dict[str, CollectionWithMapsAndStats] = {}

findstat_collection = (
    (Permutations, "Permutations", Permutation),
    (ParkingFunctions, "Parking functions", ParkingFunction),
    # (Partitions, "Integer partitions", Partition),
    # (PlanePartitions, "Plane partitions", PlanePartition),
    # (DyckWords, "Dyck paths", DyckWord),
)

for c, n, e in findstat_collection:
    print(f"Downloading data for {n}")
    maps = FindStatMaps(domain=n, codomain=n)
    maps = filter(lambda x: x.properties_raw().find("bijective") >= 0, maps)
    maps = [FindStatFunction(m) for m in maps]

    stats = FindStatStatistics(n)
    stats = [FindStatFunction(s) for s in stats]

    collection = CollectionWithMapsAndStats(c, n, stats, maps, e)
    all_collections[n] = collection

print("done downloading data")
print("printing stats...")
print(all_collections)


# Words


def get_add_n_map(n):
    def add_n(p):
        return Word([(x + n - 1) % len(p) + 1 for x in p])

    return NamedFunction(add_n, f"Add {n} to each element")


def get_mult_n_map(n):
    def mult_n(p):
        return Word([(x * n - 1) % len(p) + 1 for x in p])

    return NamedFunction(mult_n, f"Multiply each element by {n}")


complement = NamedFunction(
    lambda x: Word(list(len(x) - e + 1 for e in x)), "Complement"
)


all_collections["Words"] = CollectionWithMapsAndStats(
    lambda n: Words(n, n),
    "Words",
    [],
    [get_add_n_map(n) for n in range(1, 6)]
    + [get_add_n_map(n) for n in range(-5, 0)]
    + [get_mult_n_map(n) for n in range(1, 6)]
    + [complement],
    Word,
)


# Fubini rankings


def is_fubini_ranking(lst):
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


class FubiniRanking(HashableList[int]):
    pass


class FubiniRankings(Parent, HashableList[FubiniRanking]):
    def __init__(self, n):
        if isinstance(n, Iterable):
            HashableList.__init__(self, [FubiniRanking(x) for x in n])
        elif isinstance(n, int):
            HashableList.__init__(
                self, [FubiniRanking(x) for x in generate_fubini_rankings(n)]
            )
        else:
            raise ValueError("n must be either a list or an integer")

    def __str__(self):
        return f"Fubini ranking of size {self.n}"

    def __repr__(self):
        return f"Fubini ranking of size {self.n}"


all_collections["Fubini rankings"] = CollectionWithMapsAndStats(
    FubiniRankings,
    "Fubini rankings",
    [
        FindStatFunction.from_new_code(stat, PreMapper(stat, ParkingFunction))
        for stat in all_collections["Parking functions"].stats
    ],
    [],
    FubiniRanking,
)

print("Checking parking functions maps on Fubini rankings")
fubini_rankings = FubiniRankings(3) + FubiniRankings(4) + FubiniRankings(5)

for m in all_collections["Parking functions"].maps:
    if not does_map_project_outside_set(
        lambda x: FubiniRanking(m(ParkingFunction(x))), fubini_rankings
    ):
        all_collections["Fubini rankings"].maps.append(
            FindStatFunction.from_new_code(
                m, PostMapper(PreMapper(m, ParkingFunction), FubiniRanking)
            )
        )

print("done checking maps")

# Cayley permutations


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


class CayleyPermutation(HashableList[int]):
    pass


class CayleyPermutations(Parent, HashableList[CayleyPermutation]):
    def __init__(self, n):
        if isinstance(n, Iterable):
            HashableList.__init__(self, [CayleyPermutation(x) for x in n])
        elif isinstance(n, int):
            HashableList.__init__(
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
    CayleyPermutations,
    "Cayley permutations",
    [
        FindStatFunction.from_new_code(stat, PreMapper(stat, ParkingFunction))
        for stat in all_collections["Parking functions"].stats
    ],
    [],
    CayleyPermutation,
)


print("Checking parking functions maps on Cayley permutations")

cayley_permutations = (
    CayleyPermutations(3) + CayleyPermutations(4) + CayleyPermutations(5)
)

for m in all_collections["Parking functions"].maps:
    if not does_map_project_outside_set(m, cayley_permutations):
        all_collections["Cayley permutations"].maps.append(
            FindStatFunction.from_new_code(
                m, PostMapper(PreMapper(m, ParkingFunction), CayleyPermutation)
            )
        )

print("done checking maps")

# Homomesy result


def format_row(x, y, ins):
    if ins:
        return f"{x}: {y}"
    else:
        return f"({x}: {y})"


def print_row(avg, orbit, stats, in_set):
    t = [format_row(x, y, inss) for x, y, inss in zip(orbit, stats, in_set)]
    if all(in_set):
        print(f"{avg}:\t {' -> '.join(t)}")
    else:
        print(f"*{avg}:\t {' -> '.join(t)}")


@dataclass
class HomomesyResult:
    all_avgs: list
    all_orbits: list
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

    def homomesy_constant(self):
        if self.is_homomesic():
            return self.all_avgs[0]
        else:
            return None

    def homometric_constants_map(self):
        if self.is_homometric():
            return {
                l: a for l, a in zip([len(ll) for ll in self.all_orbits], self.all_avgs)
            }
        else:
            return None

    def is_homometric(self):
        len_queue_queue = list(set([len(c) for c in self.all_orbits]))
        is_homometric = True
        for l in len_queue_queue:
            avgs = list(
                [a for a, x in zip(self.all_avgs, self.all_orbits) if len(x) == l]
            )
            if not all([x == avgs[0] for x in avgs]):
                is_homometric = False
                break
        return is_homometric

    def is_homometric_in_set(self):
        len_queue_queue = list(set([len(c) for c in self.all_orbits]))
        is_homometric = True
        for l in len_queue_queue:
            avgs = list(
                [a for a, x in zip(self.all_avgs, self.all_orbits) if len(x) == l]
            )
            in_sets = list(
                [x for x, y in zip(self.all_in_sets, self.all_orbits) if len(y) == l]
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
                self.all_avgs, self.all_orbits, self.all_stats, self.all_in_sets
            ):
                if a == avg:
                    grouped_orbits[avg].all_avgs.append(a)
                    grouped_orbits[avg].all_orbits.append(c)
                    grouped_orbits[avg].all_stats.append(s)
                    grouped_orbits[avg].all_in_sets.append(ins)
        return grouped_orbits

    def get_all_elements(self):
        return sum(self.all_orbits, [])

    def print_all_orbits(self):
        for a, c, s, ins in zip(
            self.all_avgs, self.all_orbits, self.all_stats, self.all_in_sets
        ):
            print_row(a, c, s, ins)

    def print_result(self):
        print(f"Homomesic: {self.is_homomesic()}", end="")
        if self.is_homomesic():
            print(f" with homomesy constant {self.homomesy_constant()}")
        else:
            print()
        print(f"Homomesic using only orbits in set: {self.is_homomesic_in_set()}")
        print(f"Homometric: {self.is_homometric()}", end="")
        if self.is_homometric():
            print(f" with homometric constants {self.homometric_constants_map()}")
        else:
            print()
        print(f"Homometric using only orbits in set: {self.is_homometric_in_set()}")

    def print_grouped_orbits(self):
        for avg, res in self.get_grouped_orbits().items():
            print(f"Orbit with average {avg}")
            print()
            res.print_all_orbits()
            print()

    def print_latex_orbit_code(
        self,
        main_circle_radius=6,
        orbit_circle_radius_map={1: 0, 2: 1, 4: 2, 8: 3},
        print_function=lambda p: "$(" + ", ".join([str(e) for e in p]) + ")$",
        x_stretch=1.2,
        print_avg=True,
    ):
        main_step_size = 2 * np.pi / len(self.all_orbits)
        for i, O in enumerate(self.all_orbits):
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
                print(
                    f"\\node at ({x_center*x_stretch:.3f}, {y_center:.3f}) {{{self.all_avgs[i]}}};"
                )
            print("\n")


def lazy_orbits(S, f, yield_in_set=False):
    S = set(S)
    visited = set()
    for e in S:
        if e in visited:
            continue
        orb = [e]
        in_set = [True]
        visited.add(e)
        curr = f(e)
        while curr != e:
            if curr in orb:
                raise ValueError(
                    "Function is not a bijection. Orbit: "
                    + " -> ".join([str(x) for x in orb + [curr]])
                )
            if yield_in_set:
                in_set.append(curr in S)
            visited.add(curr)
            orb.append(curr)
            curr = f(curr)
        if yield_in_set:
            yield orb, in_set
        else:
            yield orb


def check_homomesy(S, bijection, stat):
    all_avgs = []
    all_orbits = []
    all_stats = []
    all_in_sets = []

    for orbit, in_set in lazy_orbits(S, bijection, yield_in_set=True):

        cur_orbit = list(orbit)
        cur_stats = [stat(x) for x in cur_orbit]
        cur_avg = ~(QQ(len(cur_orbit))) * sum(cur_stats)
        all_avgs.append(cur_avg)
        all_orbits.append(cur_orbit)
        all_stats.append(cur_stats)
        all_in_sets.append(in_set)

    return HomomesyResult(all_avgs, all_orbits, all_stats, all_in_sets)


def check_homomesy_with_orbits(
    orbits,
    stat,
):
    all_avgs = []
    all_stats = []
    all_in_sets = []

    for orbit in orbits:
        cur_orbit = list(orbit)
        cur_stats = [stat(x) for x in cur_orbit]
        cur_avg = ~(QQ(len(cur_orbit))) * sum(cur_stats)
        all_avgs.append(cur_avg)
        all_stats.append(cur_stats)
        all_in_sets.append([True] * len(cur_orbit))

    return HomomesyResult(all_avgs, orbits, all_stats, all_in_sets)


# def check_homomesy_for_collection_with_maps_and_stats(S, bijections, stats):
#     results = []
#     stat_values = {ele: [stat(ele) for stat in stats] for ele in S}
#     for f in bijections:
#         orbits = lazy_orbits(S, f, yield_in_set=False)
#         all_avgs_matrix = []
#         all_stats_matrix = []
#         all_in_sets = []
#         for orbit in orbits:
#             cur_orbit = list(orbit)
#             cur_stats = [stat_values[x] for x in cur_orbit]
#             cur_avgs = [~(QQ(len(cur_orbit))) * sum(x) for x in zip(*cur_stats)]
#             all_avgs_matrix.append(cur_avgs)
#             all_stats_matrix.append(cur_stats)
#             all_in_sets.append([True] * len(cur_orbit))
#         for i in range(len(stats)):
#             all_avgs = [x[i] for x in all_avgs_matrix]
#             all_stats = [x[i] for x in all_stats_matrix]
#             results.append(
#                 (
#                     f,
#                     stats[i],
#                     HomomesyResult(all_avgs, all_stats, all_stats, all_in_sets),
#                 )
#             )
#     return results


# A bunch of statistics


def the_number_of_fixed_points(lst: HashableList[int]) -> int:
    return len([e for i, e in enumerate(lst) if i + 1 == e])


the_number_of_fixed_points = FindStatFunction(
    the_number_of_fixed_points, "St000022", "Number of fixed points"
)


def first_entry(lst: HashableList[int]) -> int:
    return lst[0]


first_entry = FindStatFunction(first_entry, "St000055", "First entry of the sequence")


def the_number_of_excedances(lst: HashableList[int]) -> int:
    return sum([1 for i in range(len(lst)) if lst[i] > i + 1])


the_number_of_excedances = FindStatFunction(
    the_number_of_excedances, "St000155", "Number of excedances"
)


def the_number_of_weak_excedances(lst: HashableList[int]) -> int:
    return sum([1 for i in range(len(lst)) if lst[i] >= i + 1])


the_number_of_weak_excedances = FindStatFunction(
    the_number_of_weak_excedances, "St000213", "Number of weak excedances"
)


def the_number_of_non_cyclical_small_weak_excedances(lst: HashableList[int]) -> int:
    n = len(lst)
    return sum(1 for i in range(n) if lst[i] != ((i + 1) % n) + 1)


the_number_of_non_cyclical_small_weak_excedances = FindStatFunction(
    the_number_of_non_cyclical_small_weak_excedances,
    "St000235",
    "Number of indices that are not cyclical small weak excedances",
)


def the_number_of_cyclical_small_weak_excedances(lst: HashableList[int]) -> int:
    n = len(lst)
    return sum(1 for i in range(n) if lst[i] in [i + 1, (i + 1) % n + 1])


the_number_of_cyclical_small_weak_excedances = FindStatFunction(
    the_number_of_cyclical_small_weak_excedances,
    "St000236",
    "Number of indices that are cyclical small weak excedances",
)


def the_number_of_small_excedances(lst: HashableList[int]) -> int:
    return sum(1 for i in range(len(lst)) if lst[i] == i + 2)


the_number_of_small_excedances = FindStatFunction(
    the_number_of_small_excedances, "St000237", "Number of small excedances"
)


def the_number_of_non_small_weak_excedances(lst: HashableList[int]) -> int:
    return sum(1 for i in range(len(lst)) if lst[i] != i + 1 and lst[i] != i + 2)


the_number_of_non_small_weak_excedances = FindStatFunction(
    the_number_of_non_small_weak_excedances,
    "St000238",
    "Number of indices that are not small weak excedances",
)


def the_number_of_small_weak_excedances(lst: HashableList[int]) -> int:
    return sum(1 for i in range(len(lst)) if lst[i] in [i + 1, i + 2])


the_number_of_small_weak_excedances = FindStatFunction(
    the_number_of_small_weak_excedances, "St000239", "Number of small weak excedances"
)


def the_number_of_non_small_excedances(lst: HashableList[int]) -> int:
    return sum(1 for i in range(len(lst)) if lst[i] != i + 2)


the_number_of_non_small_excedances = FindStatFunction(
    the_number_of_non_small_excedances,
    "St000240",
    "Number of indices that are not small excedances",
)


def the_number_of_cyclical_small_excedances(lst: HashableList[int]) -> int:
    n = len(lst)
    return sum(1 for i in range(n) if lst[i] == ((i + 1) % n + 1))


the_number_of_cyclical_small_excedances = FindStatFunction(
    the_number_of_cyclical_small_excedances,
    "St000241",
    "Number of indices that are cyclical small weak excedances",
)


def the_number_of_non_cyclical_small_excedances(lst: HashableList[int]) -> int:
    n = len(lst)
    return sum(1 for i in range(n) if lst[i] not in [i + 1, (i + 1) % n + 1])


the_number_of_non_cyclical_small_excedances = FindStatFunction(
    the_number_of_non_cyclical_small_excedances,
    "St000242",
    "Number of indices that are not cyclical small weak excedances",
)


general_statistics = [
    the_number_of_fixed_points,
    first_entry,
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


for v in all_collections.values():
    for s in general_statistics:
        if s not in v.stats:
            v.stats.append(s)

print("Initialization complete. You can now use the homomesy module.")
