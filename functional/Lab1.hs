-- Exercise 0a
doubleList :: [a] -> [a]
doubleList = concatMap (\x -> [x, x])

-- Exercise 0b
firstDoubled :: Eq a => [a] -> Maybe a
firstDoubled (x:y:rest)
    | x == y = Just x
    | otherwise = firstDoubled (y:rest)
firstDoubled _ = Nothing

data Allergen = Nuts | Gluten | Soy | Dairy      deriving (Show, Eq)

type Recipe   = [Allergen]

type Name     = String
type Price    = Int
data Cupcake  = CC Name Recipe Price             deriving (Show,Eq)

r1, r2, r3, r4, r5 :: Recipe
r1 = [Gluten]
r2 = []
r3 = [Nuts]
r4 = [Dairy,Gluten]
r5 = [Soy]

onsale :: [Cupcake]
onsale = [CC "Chocolate Surprise" r1 200,
          CC "Lemon Mayhem" r2 150,
          CC "Peanut Butter Bliss" r3 150,
          CC "Yogurt Truly" r4 250,
          CC "Caramel Karma" r5 200]

-- Exercise 1a
priceRange :: Price -> Price -> [Cupcake] -> [Name]
priceRange lo hi = go
  where
    go [] = []
    go (CC name _ p : cs)
      | lo <= p && p <= hi = name : go cs
      | otherwise = go cs

-- Exercise 1b
allergyFree :: [Allergen] -> [Cupcake] -> [Name]
allergyFree bads = go
  where
    safeRecipe :: Recipe -> Bool
    safeRecipe [] = True
    safeRecipe (a:as) = notElem a bads && safeRecipe as

    go [] = []
    go (CC name recipe _ : cs)
      | safeRecipe recipe = name : go cs
      | otherwise = go cs

type Tin = [Recipe]
data Spec = And Spec Spec | Or Spec Spec | Not Spec | HasCup Int Allergen  deriving (Show,Eq)

sampletin :: Tin
sampletin = [r3,r4,r2,r5]

-- Exercise 2a
checkSpec :: Spec -> Tin -> Bool
checkSpec (And s1 s2) t = checkSpec s1 t && checkSpec s2 t
checkSpec (Or s1 s2) t = checkSpec s1 t || checkSpec s2 t
checkSpec (Not s) t = not (checkSpec s t)
checkSpec (HasCup k x) t = elem x (t !! k)

-- Exercise 2b (optional)
checkSpec' :: Spec -> Tin -> Maybe Bool
checkSpec' s t
   | wellFormed s (length t) = Just (checkSpec s t)
   | otherwise = Nothing
   where 
      wellFormed :: Spec -> Int -> Bool
      wellFormed (And s1 s2) n = wellFormed s1 n && wellFormed s2 n
      wellFormed (Or s1 s2) n = wellFormed s1 n || wellFormed s2 n
      wellFormed (Not s') n = wellFormed s' n
      wellFormed (HasCup k _) n = 0 <= k && k < n

data Tree a b = Leaf a | Node b [Tree a b]  deriving (Show,Eq)

texample :: Tree Char Integer
texample = Node 1 [Node 2 [Leaf 'a', Leaf 'b'], Node 3 [Leaf 'c', Leaf 'd', Leaf 'e'], Node 4 []]

bst :: Tree () Char
bst = Node 'c' [Node 'a' [Leaf (), Node 'b' [Leaf (), Leaf ()]], Node 'd' [Leaf (), Leaf ()]]

-- Exercise 3a
canopy :: Tree a b -> [a]
canopy (Leaf a) = [a]
canopy (Node _ cs) = forest_canopy cs
   where 
      forest_canopy [] = []
      forest_canopy (t:ts) = canopy t  ++ forest_canopy ts

-- Exercise 3b (optional)
preorder :: Tree a b -> [Either a b]
preorder (Leaf a) = [Left a]
preorder (Node b cs) = Right b : forest_pre cs
   where
       forest_pre [] = []
       forest_pre (t:ts) = preorder t ++ forest_pre ts

-- Exercise 4
linearSort :: Ord a => [a] -> [a]
linearSort xs = finalize xs [] []  -- input, stack (top = head), outRev (reverse of output)
  where
    finalize :: Ord a => [a] -> [a] -> [a] -> [a]
    finalize []     stk outRev = reverse (reverse stk ++ outRev)
    finalize (x:xs') stk outRev =
      let (toPop, stkRest) = span (\s -> x > s) stk
          outRev'          = reverse toPop ++ outRev
          stk'             = x : stkRest
      in finalize xs' stk' outRev'

-- Exercise 5a (optional)
counterexample :: [Int]
counterexample = [2,3,1]

data Bin = L | B Bin Bin  deriving (Show,Eq)

-- Exercise 5b (optional)
fromBin :: Bin -> [Int]
fromBin = error "[optional]"
toBin :: [Int] -> Maybe Bin
toBin = error "[optional"

--- Theory exercises (from Lecture 1 notes)

-- Exercise 2.2
{-
   (This is an example of a multi-line comment, delimited by
   curly braces.)
   f :: Either a (Both b c) -> Both (Either a b) (Either a c)
   g :: Both (Either a b) (Either a c) -> Either a (Both b c)
  
   f (Left a) = Pair (Left a) (Left a)
   f (Right (Pair b c)) = Pair (Right b) (Right c)

   g (Pair (Left a) (Left _)) = Left a
   g (Pari (Right b) (Right c) = Right (Pair b c)

   conclusion: they dont form an isomorphism because g cannot be total since it fails when one
   component is Left a and the other is Right _. Hence the claimed type isomorphism does not 
   hold.
-}

-- Exercise 3.1
{-
   1). [] ++ xs = xs   - holds by definition
   2). xs ++ [] = xs
	Proof:
	[] ++ [] = []
	(x:xs') ++ []
	= x:(xs'++[])
	= x:xs'
	= xs
   3). xs ++ (ys ++ zs) = (xs ++ ys) ++ zs 
	Proof:
	Base case:
	[] ++ (ys++zs) = ys++zs
	([]++ys)++zs = ys++zs

	Inductive case:
	(x:xs')
	= x:(xs'++(ys++zs))
	= x:((xs'++ys)++zs)
	= (x:(xs'++ys))++zs
	= ((x:xs')++ys)++zs
-}
-- Exercises 3.2-3.4 (optional)


