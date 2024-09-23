:- consult('database.pl').

% Вспомогательный предикат для вывода списка
print_list([]).
print_list([H|T]) :- writeln(H), print_list(T).

% Основной предикат для выполнения запросов и предоставления рекомендаций
recommend(Age, Genres) :-
    writeln('Age:'), writeln(Age),
    writeln('Genres:'), writeln(Genres),

    % Логирование проверки возраста
    (Age < 13 ->
        writeln('You are too young to play these games.');
    Age >= 13 ->
        findall(Game, 
            (member(Genre, Genres), 
             format(atom(GenreAtom), '~w', [Genre]),
             жанр(GenreAtom), 
             жанр_игры(Game, GenreAtom)),
            RecommendedGames),
        (RecommendedGames \= [] -> 
            (writeln('Recommended games:'), print_list(RecommendedGames));
            writeln('No games found for the given genres.')
        )
    ).
% Основной предикат для взаимодействия с пользователем
main :-
    write('Enter your age and preferences (e.g., "I am 13 years old, I like RPG games"): '),
    read_line_to_string(user_input, Input),
    writeln('Input:'), writeln(Input),
    parse_input(Input, Age, Genres),
    recommend(Age, Genres).

% Проверка существования жанра
genre_exists(Genre) :-
    жанр(Genre).
    
% Исправление парсинга возраста
parse_input(Input, Age, Genres) :-
    % Логируем входную строку
    writeln('Input:'), writeln(Input),

    % Разбиваем строку на части по запятой (для извлечения возраста и жанров)
    split_string(Input, ",", "", Parts),
    writeln('Parts:'), writeln(Parts),

    % Парсинг возраста
    (member(AgePart, Parts),
    writeln('Age part candidate:'), writeln(AgePart),
    sub_string(AgePart, _, _, _, "years") ->
        split_string(AgePart, " ", "", AgeWords),
        writeln('Age words:'), writeln(AgeWords),
        (nth0(2, AgeWords, AgeString) -> 
            (number_string(Age, AgeString) -> 
                writeln('Parsed age:'), writeln(Age)
            ; 
                writeln('Invalid age format'), fail
            )
        ; 
            writeln('No valid age found'), fail
        )
    ;
        writeln('Age parsing failed'), fail
    ),

    % Парсинг жанров
    member(GenresPart, Parts),
    split_string(GenresPart, " ", "", GenresWords),
    writeln('Genres words:'), writeln(GenresWords),

    % Найти индексы слов "like" и "games"
    (   nth0(LikeIndex, GenresWords, "like"),
        nth0(GamesIndex, GenresWords, "games"),
        LikeIndex < GamesIndex ->
        % Вычисляем верхнюю границу
        UpperBound is GamesIndex - 1,
        
        % Извлечение жанров
        findall(Genre, (
            between(LikeIndex, UpperBound, Index),
            nth0(Index, GenresWords, Genre),
            Genre \= "like",
            Genre \= "games"
        ), Genres),
        
        writeln('Genres list:'), writeln(Genres)
    ; 
        writeln('Genres parsing failed'), fail
    ).

:- initialization(main).
