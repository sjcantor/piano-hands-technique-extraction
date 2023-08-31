def average_annotator_per_piece(ids, match_rates):
    # compute average_annotator_per_piece
    number, _ = zip(*ids)
    mmr_all = []
    number = list(set(number))
    for n in number:
        mrs = [mr for id_piece, mr in zip(number, match_rates) if id_piece == n]
        mmr = sum(mrs) / len(mrs)
        mmr_all.append(mmr)
    # averaging all the multi-annotator match rates
    return mmr_all, number

def general_match_rate(y_pred, y_true, ids, lengths=0):
    # indicating how closely the estimation agrees with all the ground truths
    # compute match rate for every piece fingered

    if lengths == 0:
        lengths = [len(yy) for yy in y_pred]

    match_rates = []
    for p, t, l, id_piece in zip(y_pred, y_true, lengths, ids):
        assert len(p) == len(t) == l, f"id {id_piece}: apples with lemons gmr: {len(p)} != {len(t)} != {l}"
        matches = 0
        for idx, (pp, tt) in enumerate(zip(p, t)):
            if idx >= l:
                break
            else:
                if pp == tt:
                    matches += 1
        match_rates.append(matches/l)
    return average_annotator_per_piece(ids, match_rates)


def avg_general_match_rate(y_pred, y_true, ids, lengths=0):
    gmr, _ = general_match_rate(y_pred, y_true, ids, lengths=0)
    return sum(gmr) / len(gmr)