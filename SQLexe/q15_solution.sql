select city, count(city)
from teams
group by city
having count(city) > 1;

-- used template from
-- https://stackoverflow.com/questions/7151401/sql-query-for-finding-records-where-count-1
