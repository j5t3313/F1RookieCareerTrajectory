from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class OutcomeLevel(IntEnum):
    SUB_SEASON = 1
    SHORT_CAREER = 2
    MULTI_SEASON_LIMITED = 3
    ESTABLISHED_PODIUMS = 4
    RACE_WINNER = 5


@dataclass
class TeammateAssignment:
    abbreviation: str
    name: str
    start_round: int
    end_round: int


@dataclass
class RookieEntry:
    abbreviation: str
    name: str
    year: int
    team: str
    teammates: list[TeammateAssignment]
    outcome_level: Optional[OutcomeLevel]
    flags: list[str]


ROOKIE_COHORT = [
    RookieEntry(
        abbreviation="NOR",
        name="Lando Norris",
        year=2019,
        team="McLaren",
        teammates=[TeammateAssignment("SAI", "Carlos Sainz", 1, 21)],
        outcome_level=OutcomeLevel.RACE_WINNER,
        flags=[]
    ),
    RookieEntry(
        abbreviation="RUS",
        name="George Russell",
        year=2019,
        team="Williams",
        teammates=[TeammateAssignment("KUB", "Robert Kubica", 1, 21)],
        outcome_level=OutcomeLevel.RACE_WINNER,
        flags=["teammate_returning_from_injury"]
    ),
    RookieEntry(
        abbreviation="ALB",
        name="Alexander Albon",
        year=2019,
        team="Toro Rosso / Red Bull",
        teammates=[
            TeammateAssignment("KVY", "Daniil Kvyat", 1, 12),
            TeammateAssignment("VER", "Max Verstappen", 13, 21)
        ],
        outcome_level=OutcomeLevel.ESTABLISHED_PODIUMS,
        flags=["mid_season_team_change"]
    ),
    RookieEntry(
        abbreviation="LAT",
        name="Nicholas Latifi",
        year=2020,
        team="Williams",
        teammates=[TeammateAssignment("RUS", "George Russell", 1, 17)],
        outcome_level=OutcomeLevel.SHORT_CAREER,
        flags=[]
    ),
    RookieEntry(
        abbreviation="TSU",
        name="Yuki Tsunoda",
        year=2021,
        team="AlphaTauri",
        teammates=[TeammateAssignment("GAS", "Pierre Gasly", 1, 22)],
        outcome_level=OutcomeLevel.MULTI_SEASON_LIMITED,
        flags=[]
    ),
    RookieEntry(
        abbreviation="MSC",
        name="Mick Schumacher",
        year=2021,
        team="Haas",
        teammates=[TeammateAssignment("MAZ", "Nikita Mazepin", 1, 22)],
        outcome_level=OutcomeLevel.SHORT_CAREER,
        flags=["rookie_vs_rookie"]
    ),
    RookieEntry(
        abbreviation="MAZ",
        name="Nikita Mazepin",
        year=2021,
        team="Haas",
        teammates=[TeammateAssignment("MSC", "Mick Schumacher", 1, 22)],
        outcome_level=OutcomeLevel.SUB_SEASON,
        flags=["rookie_vs_rookie", "non_performance_exit"]
    ),
    RookieEntry(
        abbreviation="ZHO",
        name="Zhou Guanyu",
        year=2022,
        team="Alfa Romeo",
        teammates=[TeammateAssignment("BOT", "Valtteri Bottas", 1, 22)],
        outcome_level=OutcomeLevel.MULTI_SEASON_LIMITED,
        flags=[]
    ),
    RookieEntry(
        abbreviation="PIA",
        name="Oscar Piastri",
        year=2023,
        team="McLaren",
        teammates=[TeammateAssignment("NOR", "Lando Norris", 1, 22)],
        outcome_level=OutcomeLevel.RACE_WINNER,
        flags=[]
    ),
    RookieEntry(
        abbreviation="DEV",
        name="Nyck de Vries",
        year=2023,
        team="AlphaTauri",
        teammates=[TeammateAssignment("TSU", "Yuki Tsunoda", 1, 10)],
        outcome_level=OutcomeLevel.SUB_SEASON,
        flags=[]
    ),
    RookieEntry(
        abbreviation="SAR",
        name="Logan Sargeant",
        year=2023,
        team="Williams",
        teammates=[TeammateAssignment("ALB", "Alexander Albon", 1, 22)],
        outcome_level=OutcomeLevel.SHORT_CAREER,
        flags=[]
    ),
    RookieEntry(
        abbreviation="COL",
        name="Franco Colapinto",
        year=2024,
        team="Williams",
        teammates=[TeammateAssignment("ALB", "Alexander Albon", 15, 24)],
        outcome_level=None,
        flags=["right_censored", "partial_season_debut"]
    ),
    RookieEntry(
        abbreviation="ANT",
        name="Kimi Antonelli",
        year=2025,
        team="Mercedes",
        teammates=[TeammateAssignment("RUS", "George Russell", 1, 24)],
        outcome_level=None,
        flags=["right_censored"]
    ),
    RookieEntry(
        abbreviation="BEA",
        name="Oliver Bearman",
        year=2025,
        team="Haas",
        teammates=[TeammateAssignment("OCO", "Esteban Ocon", 1, 24)],
        outcome_level=None,
        flags=["right_censored"]
    ),
    RookieEntry(
        abbreviation="DOO",
        name="Jack Doohan",
        year=2025,
        team="Alpine",
        teammates=[TeammateAssignment("GAS", "Pierre Gasly", 1, 6)],
        outcome_level=OutcomeLevel.SUB_SEASON,
        flags=[]
    ),
    RookieEntry(
        abbreviation="HAD",
        name="Isack Hadjar",
        year=2025,
        team="Racing Bulls",
        teammates=[
            TeammateAssignment("TSU", "Yuki Tsunoda", 1, 2),
            TeammateAssignment("LAW", "Liam Lawson", 3, 24)
        ],
        outcome_level=None,
        flags=["right_censored", "sub_threshold_pairing", "rookie_vs_rookie"]
    ),
    RookieEntry(
        abbreviation="BOR",
        name="Gabriel Bortoleto",
        year=2025,
        team="Sauber",
        teammates=[TeammateAssignment("HUL", "Nico Hulkenberg", 1, 24)],
        outcome_level=None,
        flags=["right_censored"]
    ),
]


MINIMUM_RACE_THRESHOLD = 5

CACHE_DIR = "./cache"
DATA_DIR = "./data"
OUTPUT_DIR = "./output"
