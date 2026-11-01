-- Fictional catalog matching lesson_notes.md. Both Chinook databases remain unchanged.
PRAGMA foreign_keys = ON;

CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name TEXT NOT NULL);
CREATE TABLE Genre (GenreId INTEGER PRIMARY KEY, Name TEXT NOT NULL);
CREATE TABLE Album (
    AlbumId INTEGER PRIMARY KEY, Title TEXT NOT NULL, ArtistId INTEGER,
    GenreId INTEGER, ReleaseYear INTEGER, PriceUSD REAL
);
CREATE TABLE Track (
    TrackId INTEGER PRIMARY KEY, Name TEXT NOT NULL, AlbumId INTEGER,
    Milliseconds INTEGER
);

INSERT INTO Artist VALUES
    (1, 'Mira Vale'), (2, 'Noah Reed'), (3, 'Lea Moss'), (4, 'Kai Rowan');
INSERT INTO Genre VALUES
    (1, 'Harbor Jazz'), (2, 'Workshop Blues'), (3, 'Shore Ambient');
INSERT INTO Album VALUES
    (1, 'Evening Lines', 1, 1, 2018, 10.00),
    (2, 'City Lights', 2, 1, 2019, 13.00),
    (3, 'Morning Blues', 3, 2, 2020, 8.00),
    (4, 'Quiet Shores', 4, 3, 2021, 9.50);
INSERT INTO Track VALUES
    (1, 'Last Ferry', 1, 242000), (2, 'Window Seat', 1, 218000),
    (3, 'Market Steps', 2, 196000), (4, 'Neon Walk', 2, 225000),
    (5, 'First Shift', 3, 211000), (6, 'Open Door', 3, 234000),
    (7, 'Tide Glass', 4, 360000), (8, 'Slow Horizon', 4, 420000);
