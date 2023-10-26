import unittest
from deftpy.crystal_analysis import Crystal

class TestCrystalAnalysis(unittest.TestCase):

    def test_bond_dissociation_enthalpies(self):
        # Create a Crystal instance with a test crystal file
        crystal = Crystal('data/test_files/OQMD_CaTiO3_POSCAR.txt')
        
        # Test the get_bond_dissociation_enthalpies method and print the result
        bond_dissociation_enthalpies = crystal.get_bond_dissociation_enthalpies()
        print(bond_dissociation_enthalpies)

    def test_reduction_potentials(self):
        # Create a Crystal instance with a test crystal file
        crystal = Crystal('data/test_files/OQMD_CaTiO3_POSCAR.txt')
        
        # Test the get_reduction_potentials method and print the result
        reduction_potentials = crystal.get_reduction_potentials()
        print(reduction_potentials)

if __name__ == '__main__':
    unittest.main()