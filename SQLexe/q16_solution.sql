select house
from houses
where teamnum = (select teamnum from players where playername = 'Adam Ariel');