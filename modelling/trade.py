for i in range(len(niv)):
    if niv[i] > 0:
        if marketindexprice[i] <= imbalanceprice[i]:
            good_trades+=1
        else:
            bad_trades+=1
    elif niv[i] < 0:
        if marketindexprice[i] >= imbalanceprice[i]:
            good_trades+=1
        else:
            bad_trades+=1
    else:
        pass

  total_trades = good_trades+bad_trades
  accuracy = (good_trades/total_trades)*100
