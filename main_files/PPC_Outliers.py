import pickle

data_path = "./main_files/SPI_data"


low_energy_outliers = {
    "0043":(

    ),
    "0044":(

    ),
    "0045":(

    ),
    "0422":(

    ),
    "0966":(

    ),
    "0967":(

    ),
    "0970":(

    ),
    "1327":(

    ),
    "1328":(

    ),
    "1657":(

    ),
    "1658":(

    ),
    "1661":(

    ),
    "1662":(

    ),
    "1664":(

    ),
    "1667":(

    ),
    "1996":(

    ),
    "1999":(

    ),
    "2000":(
        
    ),
}

with open(f"{data_path}/low_energy_outliers.pickle", "wb") as f:
    pickle.dump(low_energy_outliers, f)