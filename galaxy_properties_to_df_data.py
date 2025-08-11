def Distance_1D(X, X_POS, BoxSize):
    '''This function takes as input a 1D 
    vector containing the positions of particles, 
    the X_POS(float) that is the position respect 
    to where we will compute the Distance and the BoxSize, 
    this function consider a periodical Box. 
    The output is a 1D vector with the same size than X 
    containing the distance from X to X_POS
    example:
        xc = Distance_1D(pos[gals,0], pos[central,0], (75000/h))
        yc = Distance_1D(pos[gals,1], pos[central,1], (75000/h))
        zc = Distance_1D(pos[gals,2], pos[central,2], (75000/h))
        # with the aim of use of the normpos
        normpos = np.sqrt(xc**2 + yc**2 + zc**2)/r200[group]
    '''
    
    D=X-X_POS
    D=np.where(D>BoxSize/2, D-BoxSize, D)
    D=np.where(D<-BoxSize/2, D+BoxSize, D)
    return D

def compute_quenching_fractions(sSFR_dict, threshold=1e-11):
    """
    Calcula la fracción de galaxias quenched por grupo.
    
    Parámetros:
        sSFR_dict: dict
            Diccionario con estructura:
                {
                    group_id1: {
                        'subhalo_id': [...],
                        'sSFR': [...],
                        'rnorm': [...]
                    },
                    ...
                }
        threshold: float
            Umbral de sSFR para considerar una galaxia como quenched.
    
    Retorna:
        dict: { group_id: QF (float) }
    """
    qf_dict = {}

    for group_id, data in sSFR_dict.items():
        ssfr_array = np.array(data['sSFR'])
        if len(ssfr_array) == 0:
            qf = np.nan  # o 0.0 si prefieres
        else:
            quenched = ssfr_array < threshold
            qf = quenched.sum() / len(ssfr_array)
        qf_dict[group_id] = qf

    return qf_dict


def compute_hist_and_diff(df, bins):
    """Funcion para calcular histograma y diferencia
    """
    
    fosil = df[df['state'] == 'fosil']['QF']
    nofosil = df[df['state'] == 'non fosil']['QF']
    
    h_fosil, _ = np.histogram(fosil, bins=bins, density=True)
    h_nofosil, _ = np.histogram(nofosil, bins=bins, density=True)
    
    return fosil, nofosil, h_fosil, h_nofosil, h_fosil - h_nofosil

def compute_cdf_and_diff(df, bins):
    fosil = df[df['state'] == 'fosil']['QF']
    nofosil = df[df['state'] == 'non fosil']['QF']
    
    # Histograma normalizado
    h_fosil, _ = np.histogram(fosil, bins=bins, density=True)
    h_nofosil, _ = np.histogram(nofosil, bins=bins, density=True)
    
    # CDF (suma acumulada de las proporciones)
    cdf_fosil = np.cumsum(h_fosil * np.diff(bins))
    cdf_nofosil = np.cumsum(h_nofosil * np.diff(bins))
    
    return fosil, nofosil, cdf_fosil, cdf_nofosil, cdf_fosil - cdf_nofosil

def binned_ssfr_fraction_stats_df(df, col_rnorm, col_ssfr, bins, threshold=1e-11):
    """
    Calcula la fracción de galaxias con sSFR > threshold por bin de Rnorm,
    junto con el error binomial.

    Parámetros
    ----------
    df : DataFrame
        DataFrame que contiene las columnas especificadas.
    col_rnorm : str
        Nombre de la columna con la distancia normalizada (Rnorm).
    col_ssfr : str
        Nombre de la columna con sSFR.
    bins : array-like
        Límites de los bins en Rnorm.
    threshold : float, opcional
        Umbral de sSFR para considerar una galaxia como activa.

    Retorna
    -------
    bin_centers : np.ndarray
        Centro de cada bin.
    fracs : np.ndarray
        Fracción de galaxias con sSFR > threshold en cada bin.
    errors : np.ndarray
        Error binomial de la fracción en cada bin.
    counts : np.ndarray
        Número total de galaxias en cada bin.
    """
    # Asignar bin a cada galaxia
    bin_indices = np.digitize(df[col_rnorm], bins)

    bin_centers = []
    fracs = []
    errors = []
    counts = []

    # Recorrer cada bin
    for i in range(1, len(bins)):
        bin_center = 0.5 * (bins[i-1] + bins[i])
        bin_data = df[bin_indices == i]
        
        total = bin_data.shape[0]
        sf_count = (bin_data[col_ssfr] > threshold).sum()
        frac = sf_count / total if total > 0 else np.nan
        error = np.sqrt(frac * (1 - frac) / total) if total > 0 else np.nan

        bin_centers.append(bin_center)
        fracs.append(frac)
        errors.append(error)
        counts.append(total)

    return (
        np.array(bin_centers),
        np.array(fracs),
        np.array(errors),
        np.array(counts)
    )