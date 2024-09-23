% Факты о видеоиграх
игра('The Witcher 3').
игра('Red Dead Redemption 2').
игра('Minecraft').
игра('Fortnite').
игра('Cyberpunk 2077').
игра('GTA V').
игра('Among Us').
игра('League of Legends').
игра('Dota 2').
игра('Counter-Strike: Global Offensive').
игра('Call of Duty: Warzone').
игра('Apex Legends').
игра('Valorant').
игра('Overwatch').
игра('World of Warcraft').
игра('Final Fantasy XIV').
игра('Elder Scrolls V: Skyrim').
игра('Dark Souls III').
игра('Hollow Knight').
игра('Celeste').

% Факты о жанрах
жанр('RPG').
жанр('Action').
жанр('Adventure').
жанр('Shooter').
жанр('Strategy').
жанр('Simulation').
жанр('Sports').
жанр('Racing').
жанр('Puzzle').
жанр('Platformer').

% Факты о жанрах игр
жанр_игры('The Witcher 3', 'RPG').
жанр_игры('Red Dead Redemption 2', 'Action').
жанр_игры('Minecraft', 'Simulation').
жанр_игры('Fortnite', 'Shooter').
жанр_игры('Cyberpunk 2077', 'RPG').
жанр_игры('GTA V', 'Action').
жанр_игры('Among Us', 'Strategy').
жанр_игры('League of Legends', 'Strategy').
жанр_игры('Dota 2', 'Strategy').
жанр_игры('Counter-Strike: Global Offensive', 'Shooter').
жанр_игры('Call of Duty: Warzone', 'Shooter').
жанр_игры('Apex Legends', 'Shooter').
жанр_игры('Valorant', 'Shooter').
жанр_игры('Overwatch', 'Shooter').
жанр_игры('World of Warcraft', 'RPG').
жанр_игры('Final Fantasy XIV', 'RPG').
жанр_игры('Elder Scrolls V: Skyrim', 'RPG').
жанр_игры('Dark Souls III', 'Action').
жанр_игры('Hollow Knight', 'Action').
жанр_игры('Celeste', 'Platformer').

% Правила для жанров
rpg_игра(Игра) :- жанр_игры(Игра, 'RPG').
shooter_игра(Игра) :- жанр_игры(Игра, 'Shooter').
action_игра(Игра) :- жанр_игры(Игра, 'Action').
strategy_игра(Игра) :- жанр_игры(Игра, 'Strategy').
platformer_игра(Игра) :- жанр_игры(Игра, 'Platformer').
