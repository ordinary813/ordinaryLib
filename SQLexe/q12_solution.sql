SELECT houses.House, players.PlayerName
FROM houses, players
WHERE houses.TeamNum = players.TeamNum
GROUP BY houses.House, players.PlayerName
ORDER BY houses.House asc, players.PlayerName asc;
