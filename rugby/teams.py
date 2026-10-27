"""National teams: canonical names, Wikipedia template codes, and a home location.

The location (lat, lon of the main rugby city or capital) drives the travel-distance and
time-zone-shift features. Successor states inherit their predecessor's record (USSR -> Russia,
Czechoslovakia -> Czechia, Yugoslavia -> Serbia, West Germany -> Germany), as World Rugby does.
"""

# code: (name, lat, lon)
TEAMS = {
    "ENG": ("England", 51.46, -0.34), "IRE": ("Ireland", 53.34, -6.23),
    "SCO": ("Scotland", 55.94, -3.24), "WAL": ("Wales", 51.48, -3.18),
    "FRA": ("France", 48.92, 2.36), "ITA": ("Italy", 41.93, 12.45),
    "NZL": ("New Zealand", -36.87, 174.74), "AUS": ("Australia", -33.85, 151.06),
    "RSA": ("South Africa", -26.23, 28.06), "ARG": ("Argentina", -34.60, -58.38),
    "JPN": ("Japan", 35.68, 139.69), "FIJ": ("Fiji", -18.14, 178.44),
    "SAM": ("Samoa", -13.83, -171.76), "TON": ("Tonga", -21.14, -175.20),
    "GEO": ("Georgia", 41.72, 44.79), "ROU": ("Romania", 44.43, 26.10),
    "ESP": ("Spain", 40.42, -3.70), "POR": ("Portugal", 38.72, -9.14),
    "RUS": ("Russia", 55.76, 37.62), "USA": ("United States", 39.74, -104.99),
    "CAN": ("Canada", 43.65, -79.38), "URU": ("Uruguay", -34.90, -56.16),
    "CHI": ("Chile", -33.45, -70.67), "NAM": ("Namibia", -22.56, 17.08),
    "BEL": ("Belgium", 50.85, 4.35), "NED": ("Netherlands", 52.37, 4.90),
    "GER": ("Germany", 50.11, 8.68), "POL": ("Poland", 52.23, 21.01),
    "HKG": ("Hong Kong", 22.32, 114.17), "KOR": ("South Korea", 37.57, 126.98),
    "BRA": ("Brazil", -23.55, -46.63), "PAR": ("Paraguay", -25.26, -57.58),
    "CZE": ("Czechia", 50.08, 14.44), "UKR": ("Ukraine", 50.45, 30.52),
    "MDA": ("Moldova", 47.01, 28.86), "SUI": ("Switzerland", 46.20, 6.14),
    "SWE": ("Sweden", 59.33, 18.07), "DEN": ("Denmark", 55.68, 12.57),
    "NOR": ("Norway", 59.91, 10.75), "FIN": ("Finland", 60.17, 24.94),
    "LTU": ("Lithuania", 54.69, 25.28), "LAT": ("Latvia", 56.95, 24.11),
    "EST": ("Estonia", 59.44, 24.75), "AND": ("Andorra", 42.51, 1.52),
    "LUX": ("Luxembourg", 49.61, 6.13), "MLT": ("Malta", 35.90, 14.51),
    "ISR": ("Israel", 32.09, 34.78), "AUT": ("Austria", 48.21, 16.37),
    "HUN": ("Hungary", 47.50, 19.04), "CRO": ("Croatia", 45.81, 15.98),
    "SLO": ("Slovenia", 46.06, 14.51), "SRB": ("Serbia", 44.79, 20.45),
    "BIH": ("Bosnia and Herzegovina", 43.86, 18.41), "MNE": ("Montenegro", 42.44, 19.26),
    "BUL": ("Bulgaria", 42.70, 23.32), "GRE": ("Greece", 37.98, 23.73),
    "CYP": ("Cyprus", 35.17, 33.36), "TUR": ("Turkey", 41.01, 28.98),
    "SVK": ("Slovakia", 48.15, 17.11), "ARM": ("Armenia", 40.18, 44.51),
    "AZE": ("Azerbaijan", 40.41, 49.87), "KAZ": ("Kazakhstan", 43.24, 76.89),
    "UZB": ("Uzbekistan", 41.30, 69.24), "KGZ": ("Kyrgyzstan", 42.87, 74.59),
    "MON": ("Monaco", 43.74, 7.42), "GIB": ("Gibraltar", 36.14, -5.35),
    "BLR": ("Belarus", 53.90, 27.57), "KOS": ("Kosovo", 42.66, 21.17),
    "ALB": ("Albania", 41.33, 19.82), "MAR": ("Morocco", 33.57, -7.59),
    "TUN": ("Tunisia", 36.81, 10.18), "ALG": ("Algeria", 36.75, 3.06),
    "EGY": ("Egypt", 30.04, 31.24), "KEN": ("Kenya", -1.29, 36.82),
    "UGA": ("Uganda", 0.35, 32.58), "ZIM": ("Zimbabwe", -17.83, 31.05),
    "ZAM": ("Zambia", -15.39, 28.32), "BOT": ("Botswana", -24.63, 25.92),
    "CIV": ("Ivory Coast", 5.36, -4.01), "SEN": ("Senegal", 14.72, -17.47),
    "CMR": ("Cameroon", 3.87, 11.52), "MAD": ("Madagascar", -18.88, 47.51),
    "NGR": ("Nigeria", 6.52, 3.38), "GHA": ("Ghana", 5.60, -0.19),
    "TOG": ("Togo", 6.13, 1.22), "MLI": ("Mali", 12.64, -8.00),
    "BEN": ("Benin", 6.37, 2.39), "BUR": ("Burkina Faso", 12.37, -1.52),
    "SWZ": ("Eswatini", -26.31, 31.14), "MRI": ("Mauritius", -20.16, 57.50),
    "NIG": ("Niger", 13.51, 2.11), "BDI": ("Burundi", -3.38, 29.36),
    "MTN": ("Mauritania", 18.08, -15.98), "RWA": ("Rwanda", -1.94, 30.06),
    "SRI": ("Sri Lanka", 6.93, 79.85), "MAS": ("Malaysia", 3.14, 101.69),
    "SIN": ("Singapore", 1.35, 103.82), "THA": ("Thailand", 13.76, 100.50),
    "TPE": ("Chinese Taipei", 25.03, 121.57), "CHN": ("China", 39.90, 116.40),
    "PHI": ("Philippines", 14.60, 120.98), "IND": ("India", 19.08, 72.88),
    "PAK": ("Pakistan", 31.55, 74.34), "UAE": ("United Arab Emirates", 25.20, 55.27),
    "IRI": ("Iran", 35.69, 51.39), "LAO": ("Laos", 17.97, 102.63),
    "QAT": ("Qatar", 25.29, 51.53), "INA": ("Indonesia", -6.21, 106.85),
    "JOR": ("Jordan", 31.95, 35.93), "LBN": ("Lebanon", 33.89, 35.50),
    "CAM": ("Cambodia", 11.56, 104.92), "BRU": ("Brunei", 4.94, 114.95),
    "NEP": ("Nepal", 27.72, 85.32), "KSA": ("Saudi Arabia", 24.71, 46.68),
    "BHR": ("Bahrain", 26.23, 50.59), "GUM": ("Guam", 13.44, 144.79),
    "PNG": ("Papua New Guinea", -9.44, 147.18), "COK": ("Cook Islands", -21.21, -159.78),
    "NIU": ("Niue", -19.05, -169.87), "SOL": ("Solomon Islands", -9.43, 159.95),
    "TAH": ("Tahiti", -17.53, -149.57), "VAN": ("Vanuatu", -17.73, 168.32),
    "NCL": ("New Caledonia", -22.27, 166.46), "COL": ("Colombia", 6.24, -75.58),
    "PER": ("Peru", -12.05, -77.04), "VEN": ("Venezuela", 10.48, -66.90),
    "MEX": ("Mexico", 19.43, -99.13), "GUY": ("Guyana", 6.80, -58.16),
    "JAM": ("Jamaica", 18.02, -76.80), "TRI": ("Trinidad and Tobago", 10.65, -61.52),
    "BAR": ("Barbados", 13.10, -59.61), "BAH": ("Bahamas", 25.05, -77.36),
    "BER": ("Bermuda", 32.29, -64.78), "CAY": ("Cayman Islands", 19.29, -81.37),
    "ECU": ("Ecuador", -0.18, -78.47), "CRC": ("Costa Rica", 9.93, -84.08),
    "GUA": ("Guatemala", 14.63, -90.51), "LCA": ("Saint Lucia", 14.01, -60.99),
    "VIN": ("Saint Vincent and the Grenadines", 13.16, -61.22),
}

# Wikipedia code variants and successor states -> canonical code.
ALIASES = {
    "ROM": "ROU", "NLD": "NED", "PRT": "POR", "DEU": "GER", "FRG": "GER", "URY": "URU",
    "CHL": "CHI", "TGA": "TON", "FJI": "FIJ", "CHE": "SUI", "SWI": "SUI", "DNK": "DEN",
    "LKA": "SRI", "MYS": "MAS", "HRV": "CRO", "BGR": "BUL", "PRY": "PAR", "SVN": "SLO",
    "ZAF": "RSA", "SAF": "RSA", "WSM": "SAM", "LVA": "LAT", "ZMB": "ZAM", "BWA": "BOT",
    "SGP": "SIN", "SER": "SRB", "SCG": "SRB", "YUG": "SRB", "MDG": "MAD", "TWN": "TPE",
    "IDN": "INA", "IRN": "IRI", "GRC": "GRE", "MCO": "MON", "SPA": "ESP", "JAP": "JPN",
    "NZ": "NZL", "IRL": "IRE", "SUN": "RUS", "URS": "RUS", "CIS": "RUS", "CSK": "CZE",
    "TGO": "TOG", "NER": "NIG", "MRT": "MTN", "BRN": "BHR", "SAU": "KSA", "KHM": "CAM",
    "VUT": "VAN", "CRI": "CRC", "SVG": "VIN", "MUS": "MRI",
}

_BY_NAME = {name.lower(): code for code, (name, _, _) in TEAMS.items()}
_BY_NAME.update({"western samoa": "SAM", "swaziland": "SWZ",
                 "hong kong china": "HKG", "georgia (country)": "GEO", "rhodesia": "ZIM",
                 "ceylon": "SRI", "fr yugoslavia": "SRB", "soviet union": "RUS",
                 "czech republic": "CZE", "usa": "USA", "korea": "KOR",
                 "united states of america": "USA"})


def canonical(code_or_name):
    """Map a Wikipedia team code or team name to a canonical code, or None if not a nation."""
    if code_or_name is None:
        return None
    key = code_or_name.strip()
    up = key.upper()
    if up in TEAMS:
        return up
    if up in ALIASES:
        return ALIASES[up]
    return _BY_NAME.get(key.lower())


def name(code):
    return TEAMS[code][0]


def code_for(team_name):
    """Look up a team by display name (case-insensitive) or code; raises KeyError if unknown."""
    c = canonical(team_name)
    if c is None:
        raise KeyError(f"unknown team {team_name!r}")
    return c


TIER1 = {"ENG", "IRE", "SCO", "WAL", "FRA", "ITA", "NZL", "AUS", "RSA", "ARG"}
