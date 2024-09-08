% Создание базы знаний и запросы


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

% Факты о разработчиках игр
разработчик_игры('The Witcher 3', 'CD Projekt Red').
разработчик_игры('Red Dead Redemption 2', 'Rockstar Games').
разработчик_игры('Minecraft', 'Mojang').
разработчик_игры('Fortnite', 'Epic Games').
разработчик_игры('Cyberpunk 2077', 'CD Projekt Red').
разработчик_игры('GTA V', 'Rockstar Games').
разработчик_игры('Among Us', 'InnerSloth').
разработчик_игры('League of Legends', 'Riot Games').
разработчик_игры('Dota 2', 'Valve').
разработчик_игры('Counter-Strike: Global Offensive', 'Valve').
разработчик_игры('Call of Duty: Warzone', 'Activision').
разработчик_игры('Apex Legends', 'Respawn Entertainment').
разработчик_игры('Valorant', 'Riot Games').
разработчик_игры('Overwatch', 'Blizzard Entertainment').
разработчик_игры('World of Warcraft', 'Blizzard Entertainment').
разработчик_игры('Final Fantasy XIV', 'Square Enix').
разработчик_игры('Elder Scrolls V: Skyrim', 'Bethesda Game Studios').
разработчик_игры('Dark Souls III', 'FromSoftware').
разработчик_игры('Hollow Knight', 'Team Cherry').
разработчик_игры('Celeste', 'Matt Makes Games').


% Правило для определения, является ли игра RPG
rpg_игра(Игра) :- жанр_игры(Игра, 'RPG').

% Правило для определения, является ли игра Shooter
shooter_игра(Игра) :- жанр_игры(Игра, 'Shooter').

% Правило для определения, является ли игра Action
action_игра(Игра) :- жанр_игры(Игра, 'Action').

% Правило для определения, является ли игра Strategy
strategy_игра(Игра) :- жанр_игры(Игра, 'Strategy').

% Правило для определения, является ли игра Platformer
platformer_игра(Игра) :- жанр_игры(Игра, 'Platformer').


% Правило для определения, разработана ли игра компанией Rockstar Games
rockstar_игра(Игра) :- разработчик_игры(Игра, 'Rockstar Games').

% Правило для определения, разработана ли игра компанией CD Projekt Red
cd_projekt_red_игра(Игра) :- разработчик_игры(Игра, 'CD Projekt Red').



:- initialization(main).
main :-
    % Проверка, является ли 'The Witcher 3' игрой
    (игра('The Witcher 3') -> writeln('The Witcher 3 is a game.'); true),

    % Проверка, является ли 'The Witcher 3' RPG игрой
    (rpg_игра('The Witcher 3') -> writeln('The Witcher 3 is an RPG game.'); true),

    % Проверка, является ли 'Minecraft' игрой
    (игра('Minecraft') -> writeln('Minecraft is a game.'); true),

    % Поиск всех RPG игр
    writeln('All RPG games:'),
    findall(Игра5, rpg_игра(Игра5), Игры4),
    print_list(Игры4),

    % Поиск всех игр, разработанных Rockstar Games
    writeln('All games developed by Rockstar Games:'),
    findall(Игра6, rockstar_игра(Игра6), Игры5),
    print_list(Игры5),

    % Поиск всех игр, разработанных CD Projekt Red
    writeln('All games developed by CD Projekt Red:'),
    findall(Игра7, cd_projekt_red_игра(Игра7), Игры6),
    print_list(Игры6),

    % Поиск всех игр, которые являются RPG и разработаны CD Projekt Red
    writeln('All RPG games developed by CD Projekt Red:'),
    findall(Игра8, (rpg_игра(Игра8), cd_projekt_red_игра(Игра8)), Игры7),
    print_list(Игры7),

    % Поиск всех игр, которые являются Shooter и не разработаны Rockstar Games
    writeln('All Shooter games not developed by Rockstar Games:'),
    findall(Игра9, (shooter_игра(Игра9), \+ rockstar_игра(Игра9)), Игры8),
    print_list(Игры8),

    % Поиск всех игр, которые являются Action и разработаны Rockstar Games
    writeln('All Action games developed by Rockstar Games:'),
    findall(Игра10, (action_игра(Игра10), rockstar_игра(Игра10)), Игры9),
    print_list(Игры9).

% Вспомогательный предикат для вывода списка
print_list([]).
print_list([H|T]) :- writeln(H), print_list(T).
