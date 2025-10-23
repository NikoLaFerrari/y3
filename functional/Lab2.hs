import Data.List

--- Zipping exercises

-- Exercise 1a
my_zip :: [a] -> [b] -> [(a,b)]
my_zip = zipWith(,)
-- my_zip (x:xs) (y:ys) = (x,y) : my_zip xs ys
-- my_zip _ _ = []

-- Exercise 1b
my_zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
my_zipWith f xs zs = map (uncurry f) (zip xs zs)
-- my_zipWith _ _ _ = []

-- Exercise 1c (optional)
my_transpose :: [[a]] -> [[a]]
my_transpose [] = []
my_transpose ([]:_) = []   -- if any row is empty, stop
my_transpose xss = map head xss : my_transpose (map tail xss)

--- Folding exercises

-- Exercise 2a
bigxor :: [Bool] -> Bool
bigxor = foldr (/=) False 

-- Exercise 2b
altsum :: Num a => [a] -> a
altsum (x:xs) = x - altsum xs
altsum _ = 0
 -- altsum = foldr (\x acc -> x - acc) 0

-- Exercise 2c
my_intersperse :: a -> [a] -> [a]
my_intersperse s = foldr step []
  where
    step x [] = []
    step x acc = x : s : acc

-- Exercise 2d
my_tails :: [a] -> [[a]]
my_tails = foldr (\x acc@(ys:_) -> (x:ys):acc) [[]]

-- Exercise 2e (optional)
my_isPrefixOf :: Eq a => [a] -> [a] -> Bool
my_isPrefixOf = undefined

-- Exercise 2f (optional)
my_dropWhile :: (a -> Bool) -> [a] -> [a]
my_dropWhile = undefined

-- Exercise 2g (optional)
-- (your proof here)

--- Difference lists

type DiffList a = [a] -> [a]

toDL :: [a] -> DiffList a
toDL xs = (xs++)

fromDL :: DiffList a -> [a]
fromDL dxs = dxs []

cons :: a -> DiffList a -> DiffList a
cons x dxs = (x:) . dxs

snoc :: DiffList a -> a -> DiffList a
snoc dxs x = dxs . (x:)

-- Exercise 3a
toDLrev :: [a] -> DiffList a
toDLrev = foldr (\x dl ->dl.(x:)) id

-- Exercise 3b
my_reverse :: [a] -> [a]
my_reverse = fromDL . toDLrev

naive_reverse :: [a] -> [a]
naive_reverse []     = []
naive_reverse (x:xs) = naive_reverse xs ++ [x]

-- Exercise 3c
-- (your explanation here)

--- Regular expression matching

data RegExp = Zero | One
            | C Char
            | Plus RegExp RegExp | Times RegExp RegExp
            | Star RegExp
  deriving (Show,Eq)

accept :: RegExp -> String -> Bool

accept e w = acc e w null

-- Exercise 4a
acc :: RegExp -> String -> (String -> Bool) -> Bool
acc Zero          w k = False
acc One           w k = k w
acc (C a)         w k = case w of 
                          (b:bs) | b == a -> k bs
                          _               -> False
acc (Plus e1 e2)  w k = acc e1 w k || acc e2 w k
acc (Times e1 e2) w k = acc e1 w (\w' -> acc e2 w' k)

-- Exercise 4b (optional)
acc (Star e)      w k = k w || acc e w (\w' -> if w' == w then False else acc (Star e) w' k)

