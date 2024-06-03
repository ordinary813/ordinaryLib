CREATE TABLE teams(
Num int,
Team varchar(25),
City varchar(25),
Established int,
Coach varchar(25),
Wins int
);

CREATE TABLE players(
PlayerID int,
PlayerName varchar(25),
TeamNum int,
Age int
);

CREATE TABLE budget(
TeamNum int,
Budget int
);

CREATE TABLE houses(
TeamNum int,
House char
);

INSERT INTO teams VALUES 
(1, 'Maccabi', 'Tel-Aviv', 1960, 'Neven Spahija', 12),
(2, 'Hapoel', 'Jerusalem', 1965, 'Dainius Adomaitis', 9),
(3, 'Maccabi', 'Haifa', 1978, 'Amit Ben-David', 6),
(4, 'Hapoel', 'Tel-Aviv', 1957, 'Dani Franko', 8),
(5, 'Hapoel', 'Galil-elion', 1972, 'Barak Peleg', 11);

INSERT INTO players VALUES 
(1, 'Gil Benny', 5, 23),
(2, 'Yoval Zossman', 1, 22),
(3, 'Iftach Ziv', 5, 26),
(4, 'Omri Kasspi', 1, 32),
(5, 'Adam Ariel', 2, 27),
(6, 'Tamir Blat', 2, 24),
(7, 'Adi Cohen Saban', 2, 27),
(8, 'Naor Sharon', 3, 26),
(9, 'Rom Gefen', 3, 27),
(10, 'Matan Naor', 4, 31);

INSERT INTO budget VALUES 
(1,3500),
(2,2100),
(3,1500),
(4,2000),
(5,1700);

INSERT INTO houses VALUES 
(1,'B'),
(2,'A'),
(3,'A'),
(4,'B'),
(5,'A');