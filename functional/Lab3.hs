module Lab3 where

--- I. Untyped lambda calculus

import Data.List

type Var = String
data LExp = V Var | A LExp LExp | L Var LExp
  deriving Show

exp1, exp2, exp3 :: LExp
exp1 = L "x" (V "x")
exp2 = L "x" (L "y" (L "z" (A (A (V "x") (V "z")) (V "y"))))
exp3 = L "a" (A (V "y") (A (V "y") (V "a")))

free :: LExp -> [Var]
free (V x)     = [x]
free (A t1 t2) = free t1 `union` free t2
free (L x t1)  = free t1 \\ [x]

-- Exercise 1a
rename :: (Var -> Var) -> LExp -> LExp
rename f (V x) = V (f x)
rename f (A e1 e2) = A (rename f e1) (rename f e2)
rename f (L x e) = L (f x) (rename f e)

-- Exercise 1b
swapvars :: (Var,Var) -> LExp -> LExp
swapvars (x,y) = rename g
  where
    g z | z == x    = y
        | z == y    = x
        | otherwise = z

-- Exercise 1c
alphaEq :: LExp -> LExp -> Bool
alphaEq (V x) (V y) = x == y
alphaEq (A e1 e2) (A f1 f2) = alphaEq e1 f1 && alphaEq e2 f2
alphaEq (L x e1) (L y e2) = alphaEq e1 (swapvars (y,x) e2)
alphaEq _ _ = False

instance Eq LExp where
  t1 == t2 = alphaEq t1 t2

-- Exercise 1d
freshFrom :: [Var] -> Var -> Var
freshFrom occ base
  | base `elem` occ = head (dropWhile (`elem` occ) (tail (iterate (++"'") base)))
  | otherwise       = base


subst :: (LExp,Var) -> LExp -> LExp
subst (d,x) (V y)
  | y == x    = d
  | otherwise = V y
subst s@(d,x) (A e1 e2) = A (subst s e1) (subst s e2)
subst s@(d,x) (L y e)
  | y == x            = L y e
  | y `elem` free d   =
      let y' = freshFrom (free d `union` free e) (y ++ "'")
          e' = rename (\v -> if v == y then y' else v) e
      in L y' (subst s e')
  | otherwise         = L y (subst s e)


--- II. STLC and principal type inference

-- Exercise 2a
{-
e1, e2_ty, e3_ty, e4_ty, e5_ty, e6_ty :: String
e1 = "forall a b. a -> b -> b"
e2 = "forall a. (a -> a) -> a -> a"
e3 = "forall a b. ((a -> a) -> b) -> b"
e4 = "forall a b. (a -> a -> b) -> a -> b"
e5 = "not typable"
e6 = "forall a b. (a -> b) -> ((a -> b) -> a) -> b"
-}


-- Exercise 2b
fn1 :: a -> b -> (a -> b -> c) -> c
fn1 x y f = f x y

fn2 :: (a -> b) -> (b -> b -> a) -> (a -> a)
fn2 f g = \x -> g (f x) (f x)

fn3 :: ([a] -> b) -> a -> b
fn3 f x = f [x]

fn4 :: ((a -> a) -> b) -> b
fn4 f = f id

-- Exercise 2c (optional)
{-
mysterylam = ??
-}


-- Exercise 2d (optional)
mysteryfn = undefined

--- III. Bidirectional typing

data Ty = TV Int | Fn Ty Ty
    deriving (Show,Eq)

data LExp' = V' Var | A' LExp' LExp' | L' Var LExp' | Ann LExp' Ty
    deriving Show

bcomp = L' "x" (L' "y" (L' "z" (A' (V' "x") (A' (V' "y") (V' "z")))))

oneid = A' (Ann (L' "f" (L' "x" (A' (V' "f") (V' "x")))) (Fn (Fn (TV 0) (TV 0)) (Fn (TV 0) (TV 0)))) (L' "x" (V' "x"))

type TyCxt = [(Var,Ty)]

check :: TyCxt -> LExp' -> Ty -> Bool
synth :: TyCxt -> LExp' -> Maybe Ty

-- Exercise 3 (optional)
check g (L' x e) (Fn a b) = check ((x,a):g) e b
check g e t = case synth g e of
  Just t' -> t' == t
  _       -> False

synth g (V' x) = lookup x g
synth g (A' e1 e2) = do
  t1 <- synth g e1
  case t1 of
    Fn a b | check g e2 a -> Just b
    _ -> Nothing
synth g (Ann e t) = if check g e t then Just t else Nothing
synth _ (L' _ _) = Nothing
