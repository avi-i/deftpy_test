""" Get crystal features for structures in Yu Kumagai's Physical Review Materials Paper """
import tarfile
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
from pymatgen.core import Composition
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error


def main():
    # Li2O
    with tarfile.open(glob("playground/kumagai_data/oxygen_vacancies_db_data/Li2O/*0.tar.gz")[0], "r:gz") as tar:
        # Read the member with the name "CONTCAR-finish"
        f = tar.extractfile("CONTCAR-finish")
        print(f.read().decode("utf-8"))

    # Get data
    df_0 = pd.read_csv("playground/kumagai_data/vacancy_formation_energy_ml/charge0.csv")  # neutral vacancies
    df_1 = pd.read_csv("playground/kumagai_data/vacancy_formation_energy_ml/charge1.csv")  # +1 charged vacancies
    df_2 = pd.read_csv("playground/kumagai_data/vacancy_formation_energy_ml/charge2.csv")  # +2 charged vacancies

    # Add charge column
    df_0["charge"] = 0
    df_1["charge"] = 1
    df_2["charge"] = 2

    # Combine dataframes
    df = pd.concat([df_0, df_1, df_2], ignore_index=True).reset_index(drop=True)

    # Remove the column named "Unnamed: 0"
    df = df.drop("Unnamed: 0", axis=1)

    # Remove non-binary compounds
    df["is_binary"] = df.formula.apply(lambda x: len(Composition(x)) == 2)
    df_binary = df.loc[df.is_binary].reset_index(drop=True)

    # Remove compounds with transition metals
    df_binary["has_transition_metal"] = df_binary.formula.apply(
        lambda x: any([el.is_transition_metal for el in Composition(x)]))
    df_binary = df_binary.loc[~df_binary.has_transition_metal].reset_index(drop=True)

    # Remove unnecessary columns
    df_plot = df_binary[["formula", "full_name", "band_gap", "formation_energy", "nn_ave_eleneg", "o2p_center_from_vbm",
                         "vacancy_formation_energy", "charge"]].reset_index(drop=True)

    # Calculate crystal reduction potentials
    n_atoms = []  # number of atoms in the compound formula
    metals = []  # metal in the compound formula
    n_metals = []  # number of metal atoms in the compound formula
    oxi_states = []  # oxidation state of the metal in the compound formula
    for _, row in df_plot.iterrows():
        formula = row["formula"]
        composition = Composition(formula)
        metal = [el.symbol for el in composition.elements if el.symbol != "O"][0]
        n_metal = composition.get_el_amt_dict()[metal]
        n_oxygen = composition.get_el_amt_dict()["O"]
        oxi_state = 2 * n_oxygen / n_metal
        n_atoms.append(composition.num_atoms)
        metals.append(metal)
        n_metals.append(n_metal)
        oxi_states.append(oxi_state)
    df_plot["n_atoms"] = n_atoms
    df_plot["metal"] = metals
    df_plot["n_metal"] = n_metals
    df_plot["oxi_state"] = oxi_states
    df_plot["vr"] = df_plot["n_atoms"] * df_plot["formation_energy"] / df_plot["n_metal"] / df_plot["oxi_state"]

    # Fit basic crystal feature model (cfm)
    fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
    for i, charge in enumerate([0, 1, 2]):
        cfm = linear_model.HuberRegressor()
        X = df_plot.loc[df_plot.charge == charge, ["vr", "band_gap"]]
        y = df_plot.loc[df_plot.charge == charge, "vacancy_formation_energy"]
        cfm.fit(X, y)
        y_pred = cfm.predict(X)

        # Plot results
        axs[i].plot(y_pred, y, "o")

        # Plot parity line
        axs[i].plot([-4, 10], [-4, 10], "--")

        # Set axis limits
        axs[i].set_xlim(-4, 10)
        axs[i].set_ylim(-4, 10)

        # Add equation
        equation = "$E_v = {:.2f} {:+.2f} V_r {:+.2f} E_g$".format(cfm.intercept_, cfm.coef_[0], cfm.coef_[1])
        axs[i].set_xlabel(equation)

        # Add MAE
        mae = mean_absolute_error(y, y_pred)
        axs[i].text(0.1, 0.9, "MAE = {:.2f} eV".format(mae), size=9, transform=axs[i].transAxes)

        # Add number of data points
        axs[i].text(0.1, 0.8, f"n = {len(y)}", size=9, transform=axs[i].transAxes)

        # Add charge as title
        axs[i].set_title(f"Charge {charge}")

        # Add y-axis label
        if i == 0:
            axs[i].set_ylabel("$E_v$ (eV)")

    plt.tight_layout()
    plt.savefig("kumagai_fit.png", dpi=300)


if __name__ == "__main__":
    main()
