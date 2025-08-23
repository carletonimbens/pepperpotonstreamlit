# utils/auth.py
import hashlib, os
from typing import Dict, List, Tuple

# 29 dances (incl. American social)
DANCE_OPTIONS: List[str] = [
    "Waltz", "Tango", "Viennese Waltz", "Foxtrot", "Quickstep",
    "Cha Cha", "Samba", "Rumba", "Paso Doble", "Jive",
    "East Coast Swing", "West Coast Swing", "Bolero", "Mambo",
    "Salsa", "Bachata", "Merengue", "Hustle", "Nightclub Two-Step",
    "Country Two-Step", "Argentine Tango", "Lindy Hop", "Kizomba",
    "Zouk", "Polka", "Peabody", "Cumbia", "Charleston", "Carolina Shag",
]

# Pastel palette name -> hex
COLOR_PALETTE: Dict[str, str] = {
    "Pink":    "#FFD1DC",
    "Orange":  "#FFD8A8",
    "Green":   "#D3F9D8",
    "Blue":    "#D0EBFF",
    "Purple":  "#E9D5FF",
    "Yellow":  "#FFF3BF",
    "Teal":    "#C3FAE8",
    "Coral":   "#FFDFD3",
    "Lavender":"#EAE2F8",
    "Mint":    "#D8F5A2",
    "Gray":    "#E9ECEF",
}

def gen_salt(n: int = 16) -> str:
    return os.urandom(n).hex()

def make_hash(username: str, dance: str, color_hex: str, salt: str) -> str:
    # hash includes username so same combo on different users hashes differently
    combo = f"{username.lower()}|{dance.lower()}|{color_hex.lower()}|{salt}"
    return hashlib.sha256(combo.encode("utf-8")).hexdigest()
