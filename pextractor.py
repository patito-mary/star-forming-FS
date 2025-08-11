import numpy as np
import h5py
import pandas as pd

class GalaxyGroupExtractor:
    def __init__(self, df_groups, subhalos, halos, h, fields_subhalos, 
                 mass_threshold=9.0, ssfr_threshold=1e-11):
        """
        df_groups : DataFrame con info de grupos (debe contener columna group_column usada en extract_group_data)
        subhalos : dict cargado con il.groupcat.LoadSubhalos
        halos : dict cargado con il.groupcat.LoadHalos
        h : parametro de Hubble
        fields_subhalos : lista de campos cargados en subhalos
        """
        self.df_groups = df_groups
        self.subhalos = subhalos
        self.halos = halos
        self.h = h
        self.fields_subhalos = fields_subhalos
        self.mass_threshold = mass_threshold
        self.ssfr_threshold = ssfr_threshold

        # variables filtradas
        self.selected_mask = None
        self.Subhalomass = None
        self.pos = None
        self.SubhaloID = None
        self.SubhaloGrNr = None

    def prepare_data(self):
        """Crea SubhaloID, filtra por masa estelar minima y guarda arrays base."""
        # crear IDs para los subhalos
        self.subhalos['SubhaloID'] = np.arange(self.subhalos['count'])

        # filtro por masa estelar
        smass = np.log10(self.subhalos['SubhaloMassType'][:, 4] / self.h) + 10
        self.selected_mask = smass >= self.mass_threshold

        # guardar propiedades clave filtradas
        self.Subhalomass = np.log10(self.subhalos['SubhaloMassType'][self.selected_mask, 4] / self.h) + 10
        self.pos = (self.subhalos['SubhaloPos'][self.selected_mask, :]) / self.h
        self.SubhaloID = self.subhalos['SubhaloID'][self.selected_mask]
        self.SubhaloGrNr = self.subhalos['SubhaloGrNr'][self.selected_mask]


    def extract_group_data(self, group_column, properties=None):
        """
        Extrae propiedades para todas las galaxias (central + satélites) de cada grupo del group_column.
        
        group_column : str
            Columna de df_groups con los IDs de grupo (GroupNumber)
        properties : list[str]
            Propiedades de subhalo a incluir (debe estar en fields_subhalos).
        """
        if properties is None:
            properties = ['SubhaloSFR']  # por defecto

        # validar propiedades
        for prop in properties:
            if prop not in self.fields_subhalos and prop != 'SubhaloID':
                raise ValueError(f"La propiedad '{prop}' no fue cargada en fields_subhalos.")

        id_groups = self.df_groups[group_column]
        r200 = (self.halos['Group_R_Crit200'][id_groups]) / self.h
        central_ids = self.halos['GroupFirstSub'][id_groups]  # IDs centrales

        FS_rnorm = {}
        FS_qf = {}

        for i, group in enumerate(id_groups):
            # subhalo ID del central
            central_sub_id = central_ids[i]

            # galaxias que pertenecen al grupo y cumplen condicion de masa
            mask_group = (self.SubhaloGrNr == group)
            group_ids = self.SubhaloID[mask_group]

            # identificar posicion de la central en los datos filtrados
            mask_central = (self.SubhaloID == central_sub_id)
            
            # posiciones relativas
            x, y, z = self.pos[mask_group, 0], self.pos[mask_group, 1], self.pos[mask_group, 2]
            xc = self._distance_1d(x, self.pos[mask_central, 0])
            yc = self._distance_1d(y, self.pos[mask_central, 1])
            zc = self._distance_1d(z, self.pos[mask_central, 2])
            normpos = np.sqrt(xc**2 + yc**2 + zc**2) / r200[i]

            # datos de salida
            group_data = []
            for j, sub_id in enumerate(group_ids):
                entry = {
                    'subhalo_id': sub_id,
                    'rnorm': normpos[j],
                    'state': 'FS'
                }
                for prop in properties:
                    if prop == 'SubhaloID':
                        entry[prop] = sub_id
                    else:
                        entry[prop] = self.subhalos[prop][self.selected_mask][mask_group][j]
                group_data.append(entry)

            FS_rnorm[group] = group_data

            # calcular QF si se tienen SFR y masa
            if 'SubhaloSFR' in properties:
                sfr = self.subhalos['SubhaloSFR'][self.selected_mask][mask_group]
                mass = self.Subhalomass[mask_group]
                ssfr = sfr / (10**mass)
                quiescent_mask = ssfr < self.ssfr_threshold
                qf = np.sum(quiescent_mask) / len(ssfr) if len(ssfr) > 0 else np.nan
                FS_qf[group] = qf
            else:
                FS_qf[group] = None

        return FS_rnorm, FS_qf

    def _distance_1d(self, coord, coord_central, box_size=75000):
        """Distancia periódica en una dimensión."""
        delta = np.abs(coord - coord_central)
        return np.minimum(delta, box_size - delta) / self.h


class SubhaloCatalogUpdater:
    def __init__(self, catalog):
        """
        catalog: puede ser
          - instancia de GalaxyGroupExtractor,
          - dict estilo il.groupcat.LoadSubhalos,
          - pd.DataFrame ya filtrado (debe contener 'SubhaloID').
        """
        self.extractor = None
        self.raw = None      # dict estilo LoadSubhalos
        self.filtered_df = None  # pd.DataFrame de los subhalos filtrados (si existe)

        # Detectar tipo
        if hasattr(catalog, 'subhalos') and hasattr(catalog, 'prepare_data'):
            # parece un GalaxyGroupExtractor
            self.extractor = catalog
            self.raw = getattr(catalog, 'subhalos', None)
            # si el extractor ya tiene selected_mask, crear DataFrame filtrado
            mask = getattr(catalog, 'selected_mask', None)
            if mask is not None and self.raw is not None:
                # armar DataFrame con los campos cargados (si fields_subhalos existe)
                fields = getattr(catalog, 'fields_subhalos', None)
                if fields is None:
                    # fallback: tomar keys del raw que sean arrays y no 'count'
                    fields = [k for k,v in self.raw.items() if isinstance(v, np.ndarray)]
                rows = {}
                for f in fields:
                    if f in self.raw:
                        rows[f] = self.raw[f][mask]
                # asegurar SubhaloID presente
                if 'SubhaloID' not in rows and 'SubhaloID' in self.raw:
                    rows['SubhaloID'] = self.raw['SubhaloID'][mask]
                self.filtered_df = pd.DataFrame(rows)
        elif isinstance(catalog, dict):
            self.raw = catalog
            self.filtered_df = None
        elif isinstance(catalog, pd.DataFrame):
            self.filtered_df = catalog.copy()
            self.raw = None
        else:
            raise TypeError("catalog must be GalaxyGroupExtractor, dict or DataFrame")

    def add_column_from_hdf5(self, hdf5_path, dataset_group, id_key, value_key, new_column_name):
        # leer HDF5
        with h5py.File(hdf5_path, 'r') as f:
            grp = f[dataset_group]
            extra_ids = np.array(grp[id_key])
            extra_vals = np.array(grp[value_key])

        # map ID -> value (mas seguro que searchsorted)
        id2val = {int(k): v for k, v in zip(extra_ids, extra_vals)}

        # si tenemos DataFrame filtrado, completar primero
        added_to_df = 0
        if self.filtered_df is not None and 'SubhaloID' in self.filtered_df.columns:
            sids = self.filtered_df['SubhaloID'].astype(int).values
            new_vals = np.array([id2val.get(int(s), np.nan) for s in sids])
            self.filtered_df[new_column_name] = new_vals
            added_to_df = np.count_nonzero(~np.isnan(new_vals))

        # actualizar raw dict tambien (llenar array del tamaño total)
        if self.raw is not None:
            raw_ids = np.array(self.raw['SubhaloID']).astype(int)
            new_arr = np.array([id2val.get(int(s), np.nan) for s in raw_ids])
            self.raw[new_column_name] = new_arr
            added_to_raw = np.count_nonzero(~np.isnan(new_arr))
        else:
            added_to_raw = 0

        # resumen
        total_matched = max(added_to_df, added_to_raw)
        total_extra = len(extra_ids)
        print(f"Column '{new_column_name}' added. matched={total_matched}, extra_rows={total_extra}, added_to_df={added_to_df}, added_to_raw={added_to_raw}")

    def get_catalog(self):
        # si existe filtered_df prefieres esa vista; si no, devuelve raw
        return self.filtered_df if self.filtered_df is not None else self.raw
