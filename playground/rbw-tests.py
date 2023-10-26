import pandas as pd
from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Species
from pymatgen.core.structure import Structure

EB = pd.read_csv("../data/Eb.csv")
VR = pd.read_csv("../data/VR.csv")


def get_crystal_bond_dissociation_enthalpies(filename: str, ) -> dict:
    """
    Get the crystal bond dissociation enthalpies for a given structure.

    Parameters
    ----------
    filename : str
        The filename of the structure file.

    Returns
    -------
    dict
        A dictionary of crystal bond dissociation enthalpies.
    """
    structure = Structure.from_file(filename)

    # Add oxidation states to structure
    structure.add_oxidation_state_by_guess()

    # Get oxygen vacancies
    vacancy_generator = VacancyGenerator()
    vacancies = vacancy_generator.get_defects(structure)
    indices = [vacancy.defect_site_index for vacancy in vacancies if vacancy.site.specie.symbol == "O"]

    # Get coordination numbers
    crystal_nn = CrystalNN()
    cn_dicts = [crystal_nn.get_cn_dict(structure, i) for i in indices]

    # Get crystal bond dissociation enthalpies
    crystal_bond_dissociation_enthalpies = {}
    for cn_dict in cn_dicts:
        species = Species(list(cn_dict.keys())[0])
        symbol = species.symbol
        oxidation_state = species.oxi_state
        crystal_bond_dissociation_enthalpy = EB.loc[(EB.elem == symbol) & (EB.os == oxidation_state), "Eb"].values[0]
        crystal_bond_dissociation_enthalpies[str(species)] = crystal_bond_dissociation_enthalpy

    return crystal_bond_dissociation_enthalpies


def get_crystal_reduction_potentials(filename: str, ) -> dict:
    """
    Get the crystal reduction potentials for a given structure.

    Parameters
    ----------
    filename : str
        The filename of the structure file.

    Returns
    -------
    dict
        A dictionary of crystal reduction potentials.
    """
    structure = Structure.from_file(filename)

    # Add oxidation states to structure
    structure.add_oxidation_state_by_guess()

    # Get oxygen vacancies
    vacancy_generator = VacancyGenerator()
    vacancies = vacancy_generator.get_defects(structure)
    indices = [vacancy.defect_site_index for vacancy in vacancies if vacancy.site.specie.symbol == "O"]

    # Get coordination numbers
    crystal_nn = CrystalNN()
    cn_dicts = [crystal_nn.get_cn_dict(structure, i) for i in indices]

    # Get crystal reduction potentials
    crystal_reduction_potentials = {}
    for cn_dict in cn_dicts:
        species = Species(list(cn_dict.keys())[0])
        symbol = species.symbol
        oxidation_state = species.oxi_state
        crystal_reduction_potential = VR[(VR.elem == symbol) & (VR.n == oxidation_state)]
        crystal_reduction_potentials[str(species)] = \
        crystal_reduction_potential.sort_values(by="m", ascending=False)["Vr"].values[0]

    return crystal_reduction_potentials


def main():
    cbdes = get_crystal_bond_dissociation_enthalpies("../data/test_files/OQMD_CaTiO3_POSCAR.txt")
    crps = get_crystal_reduction_potentials("../data/test_files/OQMD_CaTiO3_POSCAR.txt")
    print(cbdes)
    print(crps)
    print(0.1 * list(cbdes.values())[0] * 4 - 1.5 * list(crps.values())[0])


if __name__ == "__main__":
    main()
