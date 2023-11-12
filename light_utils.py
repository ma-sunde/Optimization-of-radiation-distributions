# import pandas as pd
# import numpy as np
# from PIL import Image


# def csv_to_png(csv_file, png_file):
#     """
#     # Erzeugt Graustufenbilder für LMT Lichtverteilungen
#     # Eingang ist eine .csv vom LMT ohne Achseninformationen, also A1 beginnt mit dem ersten Messwert.
#     # Dezimalzeichen ist ',' und "E" in der Excel sind zulässig.

#     Parameters:
#         csv_file: The path of the of ".csv"-file
#         png_file: The path where to save the ".png"-file

#     Returns:

#     """

#     # Auslesen der .csv
#     #my_data = pd.read_csv(csv_file, sep=";", header=None, dtype=np.float64, decimal=",")
#     my_data = pd.read_csv(csv_file, sep=",", header=None, dtype=np.float32, decimal=".")

#     count_row = my_data.shape[0]  # Gives number of rows
#     count_col = my_data.shape[1]  # Gives number of columns

#     max_value_transfer = (
#         my_data.max()
#     )  # Gibt den Maximalwert jeder Spalte in einem Series Object zurück
#     max_value = max_value_transfer.max()  # Gibt den Maximalwert zurück

#     min_value_transfer = (
#         my_data.min()
#     )  # Gibt den Minimalwert jeder Spalte in einem Series Object zurück
#     min_value = min_value_transfer.min()  # Gibt den Minimalwert zurück

#     datenarray = my_data.to_numpy()  # Konvertiert das Dataframe in ein Array

#     weight_array = np.zeros(
#         (count_row, count_col)
#     )  # Neues leeres Array der Größe des Dataframes

#     for x in range(0, count_row):  # Anzahl an Reihen
#         for y in range(0, count_col):  # Anzahl an Spalten
#             z = datenarray[x, y]
#             weight_array[x, y] = (z - min_value) / (
#                 max_value - min_value
#             )  # Schreibt in das Weight_Array den relativen Helligkeitswert zwischen 0 und 1

#     # Creates PIL image
#     img = np.zeros((count_row, count_col))
#     img = Image.fromarray(
#         np.uint8(weight_array * 255), "L"
#     )  # Erzeugt Pixelwerte zwischen 0 und 255 entsprechend des weight_array
#     img.save(png_file + "/Light_distribution_rechts.png")
#     img.save(png_file + "/Light_distribution_links.png")
#     # img.show()


# if __name__ == "__main__":
#     # Write the path of the of ".csv"-file
#     csv_file = r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_csv/Beamer_Optoma_links_2013v3.csv"

#     # Write the path of the ".png"-file + filename
#     png_file = (
#         r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_png/"
#     )

#     # Execute the "csv_to_png(csv_file, png_file)" function
#     csv_to_png(csv_file, png_file)


import pandas as pd
import numpy as np
from PIL import Image


def csv_to_png(light_df, png_file):
    """
    # Erzeugt Graustufenbilder für LMT Lichtverteilungen
    # Eingang ist eine .csv vom LMT ohne Achseninformationen, also A1 beginnt mit dem ersten Messwert.
    # Dezimalzeichen ist ',' und "E" in der Excel sind zulässig.

    Parameters:
        csv_file: The path of the of ".csv"-file
        png_file: The path where to save the ".png"-file

    Returns:

    """

    # Auslesen der .csv
    # my_data = pd.read_csv(csv_file, sep=";", header=None, dtype=np.float64, decimal=",")
    # my_data = pd.read_csv(csv_file, sep=",", header=None, dtype=np.float32, decimal=".")
    my_data = light_df

    count_row = my_data.shape[0]  # Gives number of rows
    count_col = my_data.shape[1]  # Gives number of columns

    max_value_transfer = (
        my_data.max()
    )  # Gibt den Maximalwert jeder Spalte in einem Series Object zurück
    max_value = max_value_transfer.max()  # Gibt den Maximalwert zurück

    min_value_transfer = (
        my_data.min()
    )  # Gibt den Minimalwert jeder Spalte in einem Series Object zurück
    min_value = min_value_transfer.min()  # Gibt den Minimalwert zurück

    datenarray = my_data.to_numpy()  # Konvertiert das Dataframe in ein Array

    weight_array = np.zeros(
        (count_row, count_col)
    )  # Neues leeres Array der Größe des Dataframes

    for x in range(0, count_row):  # Anzahl an Reihen
        for y in range(0, count_col):  # Anzahl an Spalten
            z = datenarray[x, y]
            weight_array[x, y] = (z - min_value) / (
                max_value - min_value
            )  # Schreibt in das Weight_Array den relativen Helligkeitswert zwischen 0 und 1

    # Creates PIL image
    img = np.zeros((count_row, count_col))
    img = Image.fromarray(
        np.uint8(weight_array * 255), "L"
    )  # Erzeugt Pixelwerte zwischen 0 und 255 entsprechend des weight_array
    img.save(png_file + "/Light_distribution_rechts.png")
    img.save(png_file + "/Light_distribution_links.png")
    # img.show()


if __name__ == "__main__":
    # Write the path of the of ".csv"-file
    csv_file = r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_csv/Beamer_Optoma_links_2013v3.csv"

    # Write the path of the ".png"-file + filename
    png_file = (
        r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_png/"
    )

    # Execute the "csv_to_png(csv_file, png_file)" function
    csv_to_png(csv_file, png_file)
