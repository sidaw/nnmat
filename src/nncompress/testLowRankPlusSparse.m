Wlr0 = randn(30,3) * randn(3, 30);
Wsparse0base = [1 3 1 
                2 4 1 
                30 30,2];
Wsparse0 = spconvert(Wsparse0base);
Wtest = Wlr0 + Wsparse0;
[Wlr, Wsparse] = getLowRankPlusSparse(Wtest, 2);