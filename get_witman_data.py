""" Get crystal features for structures in Matt Witman's Nature Computational Science Paper """
from glob import glob

import adjustText
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pymatgen.core import Structure, Composition
from sklearn.linear_model import HuberRegressor
from tqdm import tqdm

from deftpy.crystal_analysis import Crystal


def main():
    data_path = "/Users/isakov/Desktop/Fall_23/structures_production/data_01_03_22/"
    #csv_paths = sorted(glob("playground/witman_data/data_01_03_22/csvs/*.csv"))
    csv_paths = sorted(glob(data_path + "csvs/*.csv"))
    #poscar_path = "playground/witman_data/data_01_03_22/poscars"
    poscar_path = data_path + "poscars"
    # Read in all the data and concatenate it into one dataframe

    # Create a column for the csv file name, not including the path and .csv extension
    df = pd.concat(
        [pd.read_csv(csv_path).assign(filename=csv_path.split("/")[-1][:-4]) for csv_path in csv_paths]
    )

    # Create a column for the defectid, which is the string after the la st period in the filename
    df["defectid"] = df["filename"].str.split(".").str[-1]

    # Drop rows with defectname != "V_O"
    df = df[df["defectname"] == "V_O"]

    # Reset the index
    df = df.reset_index(drop=True)

    # Create a column for the poscar
    # The poscars are in poscar_path, and the filename is filename + "_POSCAR_wyck"
    df["poscar"] = df["filename"].apply(lambda x: open(f"{poscar_path}/{x}_POSCAR_wyck").read())

    # Create a column for the pymatgen structure
    df["structure"] = df["poscar"].apply(lambda x: Structure.from_str(x, fmt="poscar"))

    # Add oxidation states to the structure
    for _, row in df.iterrows():
        #oxstate_path = f"playground/witman_data/data_01_03_22/oxstate/{row['filename']}_oxstate"
        oxstate_path = data_path + f"oxstate/{row['filename']}_oxstate"
        oxstate = []
        for x in open(oxstate_path).read().split():
            oxstate += int(x.split("*")[0]) * [float(x.split("*")[1])]
        structure = row["structure"]
        structure.add_oxidation_state_by_site(oxstate)

    # Group by 'formula' and find unique defect IDs and their minimum dH_ev_per_atom (1,2,3)
    #dH_eV_per_atom_min = df.groupby(['formula'])['dH_eV_per_atom'].min().reset_index()
    #dH_eV_per_atom_min = df.groupby(['formula', 'defectid'])['dH_eV_per_atom'].min().reset_index()
    dH_eV_per_atom_min = df.groupby('formula')['defectid'].unique().apply(lambda x: min(df[df['defectid'].isin(x)]['dH_eV_per_atom']))
    min_dH_df = pd.DataFrame({'formula': dH_eV_per_atom_min.index, 'min_dH': dH_eV_per_atom_min.values})

    # Merge the unique defect IDs and minimum dH back to the original DataFrame (1,3)
    #df = pd.merge(df, dH_eV_per_atom_min, on=['formula'], how='left', suffixes=('', '_min'))
    df = pd.merge(df, min_dH_df, on='formula', how='left')

    # Calculate adjusted_dH using dH_ev_per_atom and min_dH_ev_per_atom (1,3)
    #df['adjusted_dH'] = df['dH_eV_per_atom'] - df['dH_eV_per_atom_min']
    df['adjusted_dH'] = df['dH_eV_per_atom'] - df['min_dH']

    #df[['formula','defectid','dH_eV_per_atom', 'dH_eV_per_atom_min', 'adjusted_dH']].to_csv("Ehull.csv")

    # Drop the intermediate columns used for calculation if needed (1)
    #df.drop(columns=['dH_eV_per_atom_min'], inplace=True)

    # Binary?
    #df["is_binary"] = df["formula"].apply(lambda x: len(Composition(x).elements) == 2)
    # Ternary?
    df["is_binary_or_ternary"] = df["formula"].apply(lambda x: 2 <= len(Composition(x).elements) <= 3)
    #test to see if all had been included via composition elements seen in fit4
    #df["is_binary_or_ternary"] = df["formula"].apply(lambda x: 2 <= len(Composition(x).elements) <= 4)
    # Sort by defectid and then by site
    df = df.sort_values(["defectid", "site"])
    df.to_csv("oxygen_vacancies.csv", index=False)

    # Calculate crystal features for binary structures
    #df = df[df["is_binary"]]
    df = df[df["is_binary_or_ternary"]]
    df_cf = pd.DataFrame()
    for defectid in tqdm(df["defectid"].unique()):
        df_defectid = df[df["defectid"] == defectid]
        structure = df_defectid["structure"].iloc[0]
        crystal = Crystal(pymatgen_structure=structure)

        CN = crystal.cn_dicts
        Eb = crystal.bond_dissociation_enthalpies
        Vr = crystal.reduction_potentials

        # Calculate CN-weighted Eb sum
        Eb_sum = []
        for CN_dict, Eb_dict in zip(CN, Eb):
            CN_array = np.array(list(CN_dict.values()))
            Eb_array = np.array(list(Eb_dict.values()))
            Eb_sum.append(np.sum(CN_array * Eb_array))

        # Calculate maximum Vr
        Vr_max = []
        for Vr_dict in Vr:
            try:
                Vr_max.append(max(Vr_dict.values()))
            except ValueError:
                Vr_max.append(np.nan)

        # Make a dataframe
        formula = df_defectid["formula"].values
        defectid = df_defectid["defectid"].values
        site = df_defectid["site"].values
        Eg = df_defectid["bandgap_eV"].values
        Ev = df_defectid["dH_eV"].values
        Ehull = df_defectid["adjusted_dH"].values
        try:
            df_cf = pd.concat(
                [
                    df_cf,
                    pd.DataFrame(
                        {
                            "formula": formula,
                            "defectid": defectid,
                            "site": site,
                            "Eb_sum": Eb_sum,
                            "Vr_max": Vr_max,
                            "Eg": Eg,
                            "Ev": Ev,
                            "Ehull": Ehull,
                        }
                    ),
                ]
            )
        except ValueError:
            pass
    df_cf = df_cf.reset_index(drop=True)
    #df_cf.to_csv("witman_data_w_Ehull.csv", index=False)

    # plot witman-based cfm
    # remove NaNs
    df_cf = df_cf.dropna()
    cfm = HuberRegressor()
    #X = df_cf[["Vr_max", "Eg"]]
    #X = df_cf[["Eb_sum", "Vr_max", "Eg"]]
    X = df_cf[["Eb_sum", "Vr_max", "Eg", "Ehull"]]
    y = df_cf["Ev"]
    cfm.fit(X, y)
    y_pred = cfm.predict(X)
    coefs = cfm.coef_
    print(coefs)
    exit(4)
    df_cf['y_pred'] = y_pred

    unique_formulas = df_cf['formula'].unique()
    num_unique_formulas = len(unique_formulas)
    colors = plt.cm.plasma(np.linspace(0, 1, num_unique_formulas))
    #plt.scatter(y, y_pred)
    for i, formula in enumerate(unique_formulas):
        formula_data = df_cf[df_cf['formula'] == formula]
        plt.scatter(formula_data["Ev"], formula_data["y_pred"], color=colors[i], label=formula)
    plt.plot([1, 9], [1, 9], "k--")
    #equation = f"$E_v$ = {cfm.intercept_:.2f} + {cfm.coef_[0]:.2f} $V_r$ + {cfm.coef_[1]:.2f} $E_g$"
    #equation = f"$E_v$ = {cfm.intercept_:.2f} + {cfm.coef_[0]:.2f} $\\Sigma E_b$ + {cfm.coef_[1]:.2f} $V_r$ + {cfm.coef_[2]:.2f} $E_g$"
    equation = f"$E_v$ = {cfm.intercept_:.2f} + {cfm.coef_[0]:.2f} $\\Sigma E_b$ + {cfm.coef_[1]:.2f} $V_r$ + {cfm.coef_[2]:.2f} $E_g$ + {cfm.coef_[3]:.2f} $E_h$"
    plt.text(1, 9, equation, fontsize=9)
    mae = np.mean(np.abs(y - y_pred))
    plt.text(1, 8, f"MAE = {mae:.2f} eV", fontsize=9)
    # add number of data points as text
    plt.text(1, 7, f"n = {len(y)}", fontsize=9)
    # texts = []
    # for x, y, s in zip(y, y_pred, df_cf["formula"]):
    #     texts.append(plt.text(x, y, s, size=6))
    # adjustText.adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))
    plt.title("CFM for neutral oxygen vacancy formation")
    plt.xlabel(str(equation))
    plt.ylabel(f"$E_v$")
    plt.legend(bbox_to_anchor=(1.1, 1.3), prop={'size': 3.5})
    plt.savefig("witman_fit_dH_3.png", dpi=300)


if __name__ == "__main__":
    main()
