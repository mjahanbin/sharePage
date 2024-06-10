# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 21:42:20 2024

"""

import math
import numpy as np

__version__ = '0.0.1'


#%% macro defintion and constants

INVALID_INDEX = -1
ERROR_THE_LENGTH_OF_HISTORY_MUST_BE_POSITIVE = 1
ERROR_THE_NUMBER_OF_TS_MUST_BE_POSITIVE = 2
ERROR_THEORETICALLY_IMPOSSIBLE_VALUE = 3
ERROR_INCOMPLETE_PROXY = 4
ERROR_BLANK_TIME_SERIES = 5
ERROR_HISTORY_LENGTH_UNDER_THREE = 6
ERROR_NUMBER_OF_TS_UNDER_TWO = 7
ERROR_MUST_HAVE_SOME_COMPLETE_TS_FIRST = 8
ERROR_MISSING_DATA_IN_SHORT_HISTORY = 9
ERROR_DATA_IN_THE_MISSING_RECTANGLE = 10
ERROR_TOO_MANY_OR_TOO_FEW_COMPLETE_TS = 11
ERROR_TOO_LARGE_OR_TOO_SMALL_MISSING_PORTION = 12

NTS = 2
MISSING_VALUE = 1.234e-10
CHECKING_VAL = 0
CHECKING_ADJ = 1
N_CHECKS = 2
LARGE_NUMBER = 10000000.0

PARTNER = lambda p: NTS - 1 - p
MIN = min
MAX = max

from collections import namedtuple
Violator = namedtuple('Violator', [])

#%% Convert the struct to a Python class or named tuple.

from typing import List, NamedTuple

class ReportedList(NamedTuple):
    tm: int
    instr: int
    category: int
    severity: float
    value: float

class TsPoint(NamedTuple):
    head: int
    tail: int
    val: float
    adj: float

class PairStats(NamedTuple):
    left_point: int
    rght_point: int
    estimation_strength: float
    variance: List[float]
    covariance: float
    delta: float
    linear: float
    quadratic: float

"""
/* 1 */
/********************************************************
interval() - Set the intervals for a singe time series
********************************************************/
"""
def interval(r: List[TsPoint], frst: int, last: int):
    u = r[frst]
    v = r[last]
    u = u._replace(head=frst - 1)
    v = v._replace(tail=last if v.val != MISSING_VALUE else last + 1)
    
    for i in range(frst, last):
        q = r[i]
        r[i + 1] = r[i + 1]._replace(head=(frst - 1 if q.adj == MISSING_VALUE else (q.head if q.val == MISSING_VALUE else i)))
    
    for i in range(last - 1, frst - 1, -1):
        q = r[i]
        r[i] = q._replace(tail=(i if q.val != MISSING_VALUE else (r[i + 1].tail if q.adj != MISSING_VALUE else last + 1)))
    
    return r


#%%
"""
/* 2 */
/************************************************************************************************************
alessandro_palandri() - For a time series pair, remove the points that do not belong to their common history
************************************************************************************************************/

1. Function Parameters: The parameters in Python are lists or other suitable collections to allow for mutable changes.
2. Immutability Handling: Used _replace method for NamedTuple to handle immutability.
3. Indexing: Replaced pointer arithmetic with Python list indexing.
"""
def alessandro_palandri(c, d, r, q, sta_day, end_day, pstrong, pcov, variance, stats):
    """
    Compute Alessandro Palandri's shock sign restrictions.

    Args:
        c: First time series.
        d: Second time series.
        r: Response of the first time series.
        q: Response of the second time series.
        sta_day: Start day of the analysis.
        end_day: End day of the analysis.
        pstrong: Estimation strength.
        pcov: Covariance.
        variance: Variance.
        stats: Statistics.

    Returns:
        None.
    """
    
    for p in range(NTS):
        if p:
            a, b, x = d, c, q
        else:
            a, b, x = c, d, r
        for k in range(sta_day, end_day + 1):
            x[k] = x[k]._replace(val=a[k].val, adj=a[k].adj)
            if x[k].val != MISSING_VALUE:
                my_nearest_right = end_day + 1 if k == end_day or a[k + 1].head != k else a[k + 1].tail
                if my_nearest_right <= b[k].tail and a[k].head >= b[k].head:
                    x[k] = x[k]._replace(val=MISSING_VALUE)
        interval(x, sta_day, end_day)

    pcov[0] = variance[1] = variance[0] = pstrong[0] = 0.0

    for k in range(sta_day, end_day + 1):
        s = stats[k]
        s = s._replace(left_point=sta_day - 1, rght_point=end_day + 1, linear=0.0, quadratic=0.0, delta=0.0, covariance=0.0,
                       estimation_strength=0.0, variance=[0.0, 0.0])
        stats[k] = s

    for k in range(sta_day, end_day + 1):
        s = stats[k]
        left_point = min(r[k].head, q[k].head)
        rght_point = max(r[k].tail, q[k].tail)
        if left_point >= sta_day and rght_point <= end_day and min(r[k].tail, q[k].tail) == k:
            cur_overlap = k - max(r[k].head, q[k].head)
            shock = [0] * NTS
            span = [0] * NTS
            for p in range(NTS):
                t = q if p else r
                v = t[k]
                shock[p] = t[v.tail].val - t[v.head].val
                span[p] = v.tail - v.head
            s = s._replace(left_point=left_point, rght_point=rght_point)
            s.estimation_strength = (cur_overlap / span[0]) * (cur_overlap / span[1])
            pstrong[0] += s.estimation_strength
            coeff = s.estimation_strength * shock[1] / cur_overlap
            stats[r[k].tail].delta += coeff
            stats[r[k].head].delta -= coeff
            s.covariance = shock[0] * coeff
            pcov[0] += s.covariance
            for p in range(NTS):
                s.variance[p] = s.estimation_strength * shock[p] * shock[p] / span[p]
                variance[p] += s.variance[p]
            coeff = s.estimation_strength / span[0]
            v = r[k]
            stats[r[k].tail].linear += 2.0 * coeff * shock[0]
            stats[r[k].head].linear -= 2.0 * coeff * shock[0]
            stats[r[k].tail].quadratic += coeff
            stats[r[k].head].quadratic += coeff
    return


#%%
"""
/* 3 */
/***************************************************************************
corr_value() - Given covariance and the variances, returns the correlation.
***************************************************************************/
"""

def corr_value(cov, variance):
    xtemp = variance[0] * variance[1]
    return cov / math.sqrt(xtemp) if xtemp > 0.0 else 0.0

"""
/* 4 */
/***************************************************************************
suspect_analyzer() - identifies observations that are likely to be errors.
***************************************************************************/
"""
def suspect_analyzer(n_series, n_days, n_list, raw_data, adj_data, val_score, adj_score, cor_data, dt, ts, tp):
    """
    Analyzes time series data to identify suspicious points based on their impact on correlation.

    Args:
        n_series: Number of time series.
        n_days: Number of days in each time series.
        n_list: Number of top violators to report.
        raw_data: Raw time series data.
        adj_data: Adjusted time series data.
        val_score: Array to store value impact scores.
        adj_score: Array to store adjustment factor impact scores.
        cor_data: Array to store correlation data.
        dt: Array to store time indices of violators.
        ts: Array to store time series indices of violators.
        tp: Array to store violation types of violators.

    Returns:
        Version of the suspect analyzer.
    """
    if n_days < 1:
        return ERROR_THE_LENGTH_OF_HISTORY_MUST_BE_POSITIVE
    if n_series < 1:
        return ERROR_THE_NUMBER_OF_TS_MUST_BE_POSITIVE

    # Allocate memory
    estimation_strength, cov  = 0.0, 0.0
    added_strength, added_cov = 0.0, 0.0
    
    projvar   = np.zeros(NTS)
    added_var = np.zeros(NTS)
    
    added_stats = [PairStats(0, 0, 0.0, [0.0, 0.0], 0.0, 0.0, 0.0) for _ in range(n_days)]
    stats = [PairStats(0, 0, 0.0, [0.0, 0.0], 0.0, 0.0, 0.0) for _ in range(n_days)]
    added_stats = [PairStats(0, 0, 0.0, [0.0, 0.0], 0.0, 0.0, 0.0) for _ in range(n_days)]
    violators = [ReportedList(0, 0, 0, 0.0, 0.0) for _ in range(n_list)]
    modified_ts = [TsPoint(0, 0, 0.0, 0.0) for _ in range(n_days)]
    f = [[TsPoint(0, 0, 0.0, 0.0) for _ in range(n_days)] for _ in range(NTS)]
    y = [[TsPoint(0, 0, 0.0, 0.0) for _ in range(n_days)] for _ in range(NTS)]
    z = [[TsPoint(0, 0, 0.0, 0.0) for _ in range(n_days)] for _ in range(n_series)]

    for j in range(n_series):
        for k in range(n_days):
            val_score[k * n_series + j] = 0.0
            adj_score[k * n_series + j] = 0.0
            z[j][k] = TsPoint(0, 0, raw_data[k * n_series + j], adj_data[k * n_series + j])
        interval(z[j], 0, n_days - 1)
        

    for i in range(n_series):
        for j in range(i + 1):
            # Set up pillars by removing the points with zero information content (per Alessandro Palandri)
            alessandro_palandri(z[i], z[j], f[0], f[1], 0, n_days - 1, estimation_strength, cov, projvar, stats)
            corr = corr_value(cov, projvar)
            if j == i:
                ts_involved = 1
                cor_data[i * n_series + j] = math.sqrt(projvar[0] / estimation_strength) if estimation_strength > 0.0 else 0.0
            else:
                ts_involved = NTS
                cor_data[i * n_series + j] = cor_data[j * n_series + i] = corr
            for p in range(ts_involved):  # walk through the history of this pair of TS
                a = j if p else i
                b = i if p else j
                r = z[a]
                q = z[b]
                r_palandri = f[p]
                q_palandri = f[PARTNER(p)]
                # a point with missing Palandri value can only have the adj factor impact, on the condition of being inside a Palandri interval
                for k in range(n_days):
                    if not (r_palandri[k].val != MISSING_VALUE or (r[k].val != MISSING_VALUE and r_palandri[k].head >= 0 and r_palandri[k].tail < n_days)):
                        continue
                    my_start = r_palandri[k].head
                    if my_start < 0:
                        my_start = r[k].head
                    if my_start < 0:
                        my_start = k
                    if k < n_days - 1:
                        my_end = r_palandri[k + 1].tail
                        if my_end > n_days - 1:
                            my_end = r[k + 1].tail
                        if my_end > n_days - 1:
                            my_end = k
                    else:
                        my_end = k
                    if my_end > my_start:
                        partner_start = q_palandri[my_start + 1].head
                        if partner_start < 0:
                            partner_start = k
                        partner_end = q_palandri[my_end].tail
                        if partner_end > n_days - 1:
                            partner_end = k
                        common_start = min(my_start, partner_start)
                        common_end = max(my_end, partner_end)
                        local_strength = local_cov = local_var = [0.0, 0.0]
                        for d in range(common_start, common_end + 1):
                            modified_ts[d].val = r[d].val
                            modified_ts[d].adj = r[d].adj
                            if stats[d].left_point >= common_start and stats[d].rght_point <= common_end:
                                local_strength += stats[d].estimation_strength
                                local_cov += stats[d].covariance
                                local_var[0] += stats[d].variance[p]
                                local_var[1] += stats[d].variance[PARTNER(p)]
                        for w in range(N_CHECKS):  # THE INNERMOST LOOP: finding two impacts, from missing value and from missing adj factor
                            if w == CHECKING_VAL:
                                if r_palandri[k].val == MISSING_VALUE:
                                    continue  # point with missing Palandri value does not have value impact
                                xtemp = modified_ts[k].val
                                modified_ts[k].val = MISSING_VALUE
                            else:  # checking adj factor
                                if r[k].adj == MISSING_VALUE or my_end == k:
                                    continue  # guaranteed to have zero adj factor impact
                                xtemp = modified_ts[k].adj
                                modified_ts[k].adj = MISSING_VALUE
                            interval(modified_ts, common_start, common_end)
                            alessandro_palandri(modified_ts, q, y[0], y[1], common_start, common_end, added_strength, added_cov, added_var, added_stats)
                            new_cov = cov - local_cov + added_cov
                            new_var = [projvar[d] - local_var[d] + added_var[d] for d in range(NTS)]
                            corr_without = corr_value(new_cov, new_var)
                            if ts_involved == 1:  # the impact = reduction in volatility
                                variance_without = new_var[0] * (estimation_strength / (estimation_strength - 1.0) if estimation_strength > 1.0 else 1.0)
                                if projvar[0] > variance_without:  # impact > 0
                                    impact = 1.0 - math.sqrt(variance_without / projvar[0])
                                elif projvar[0] < variance_without:  # impact < 0
                                    impact = math.sqrt(projvar[0] / variance_without) - 1.0
                                else:
                                    impact = 0.0
                            else:  # the measure is average improvement in squared correlation, in percentage points
                                impact = (corr_without * corr_without - corr * corr)
                            impact *= 100.0 / n_series  # 100.0 multiplier is for display purposes
                            if w == CHECKING_VAL:
                                modified_ts[k].val = xtemp  # restore the value
                                val_score[k * n_series + a] += impact
                            else:  # checking adj factor
                                modified_ts[k].adj = xtemp  # restore the adj factor
                                adj_score[k * n_series + a] += impact
                        # two checks done, for the value and for the adjustment factor
                    # if my_end > my_start
                # the point impact has been determined
            # next point to check in the TS pair group
        # next TS pair
    # next TS pair

    n_violators = 0
    for j in range(n_series):
        for m in range(n_days):
            for w in range(N_CHECKS):
                current_severity = val_score[m * n_series + j] if w == CHECKING_VAL else adj_score[m * n_series + j]
                if n_violators == n_list and current_severity <= violators[n_list - 1].severity:
                    continue  # does not go on the list
                for k in range(n_violators):
                    if current_severity > violators[k].severity:
                        break
                if n_violators < n_list:
                    n_violators += 1
                for i in range(n_violators - 1, k, -1):
                    violators[i] = violators[i - 1]
                violators[k] = violators[k]._replace(tm=m, instr=j, severity=current_severity, category=w)
    
    for i in range(n_list):
        dt[i] = violators[i].tm
        ts[i] = violators[i].instr
        tp[i] = violators[i].category


    return __version__

"""
/* 5 */
/*************************************
about() - returns the current version.
*************************************/
"""
def about():
	return __version__

"""
/* 6 */
/***************************************************************************
goal_function() - function value for a time series pair, for a given guess.
***************************************************************************/
"""
def goal_function(i_series, n_series, a, b, delta, linear, quadratic, strength_of_inference, v, o, c, x, importance_of_variance):
    """
    Calculates the goal function for a given time series.

    Args:
        i_series: The index of the time series being evaluated.
        n_series: The total number of time series.
        a: A list of intercept terms for the correlation without the point.
        b: A list of slope terms for the correlation without the point.
        delta: A list of change in correlation due to the point.
        linear: A list of linear coefficients for the expected variance.
        quadratic: A list of quadratic coefficients for the expected variance.
        strength_of_inference: A list of strength of inference values.
        v: A list of variances for each time series.
        o: A list of variances for the other time series.
        c: A list of correlations between the time series and the other time series.
        x: The value of the point being evaluated.
        importance_of_variance: The importance of variance in the goal function.

    Returns:
        The value of the goal function.
    """
    
    goal = 0.0
    for j in range(n_series):
        if strength_of_inference[j] <= 0.0:
            continue  # no strength of inference, no information
        if o[j] <= 0.0:
            continue  # the other TS shows no variability
        my_expected_variance = v[j] + linear[j] * x + quadratic[j] * x * x
        if my_expected_variance <= 0.0:
            continue
        if i_series == j:  # against itself; check the variance
            z = math.sqrt(my_expected_variance / strength_of_inference[j])
            w = a[j] * (1.0 / (1.0 + b[j]) if b[j] <= 0.0 else 1.0 - b[j])  # statistic without that point
            if w <= 0.0:
                continue
            goal += strength_of_inference[j] * 0.5 * importance_of_variance * math.log(z / w)
        else:
            correlation_without = a[j] + b[j]
            correlation_with = (c[j] + delta[j] * x) / math.sqrt(o[j] * my_expected_variance)
            goal += strength_of_inference[j] * abs((correlation_without - correlation_with) * correlation_without)
    return goal

"""
/* 7 */
/***************************************************************************
golden_ratio_search() - find the minimum of the goal_function().
***************************************************************************/
"""
def golden_ratio_search(i_series, n_series, a, b, delta, linear, quadratic, strength_of_inference, v, o, c, goal_f, min_val, max_val):
    """
    Performs a golden ratio search to find the optimal value of a parameter.

    Args:
        i_series: The time series of infections.
        n_series: The time series of non-pharmaceutical interventions.
        a: A parameter.
        b: Another parameter.
        delta: Yet another parameter.
        linear: A boolean indicating whether to use a linear model.
        quadratic: A boolean indicating whether to use a quadratic model.
        strength_of_inference: The strength of inference.
        v: A parameter.
        o: Another parameter.
        c: Yet another parameter.
        goal_f: A list containing the importance of variance and the best suggestion.
        min_val: The minimum value of the parameter to search over.
        max_val: The maximum value of the parameter to search over.

    Returns:
        The optimal value of the parameter.
    """
    
    importance_of_variance = goal_f[0]  # the first element of the goal_f[] on input contains the importance of variance
    # importance_of_variance = 1.0  # in case we want to keep it at 1.0
    gr = (math.sqrt(5.0) + 1.0) / 2.0
    grid_element = (max_val - min_val) / (LARGE_NUMBER / 100.)  # rough estimate by scanning the historical range of values
    best_suggestion = min_val
    prev_val = goal_function(i_series, n_series, a, b, delta, linear, quadratic, strength_of_inference, v, o, c, min_val, importance_of_variance)

    for t in range(min_val, max_val, grid_element):
        new_val = goal_function(i_series, n_series, a, b, delta, linear, quadratic, strength_of_inference, v, o, c, t, importance_of_variance)
        if new_val < prev_val:
            prev_val = new_val
            best_suggestion = t

    y = [goal_function(i_series, n_series, a, b, delta, linear, quadratic, strength_of_inference, v, o, c, best_suggestion + grid_element * (m - 0.5), importance_of_variance) for m in range(2)]
    step = grid_element * (-0.5 if y[0] < y[1] else 0.5)

    t = best_suggestion + step
    prev_val = min(y)
    while abs(step) < LARGE_NUMBER:
        new_val = goal_function(i_series, n_series, a, b, delta, linear, quadratic, strength_of_inference, v, o, c, t, importance_of_variance)
        if new_val > prev_val:
            break
        prev_val = new_val
        step *= 2.
        t += step

    if step < 0:
        pa = t
        pb = best_suggestion + grid_element / 2.
    else:
        pa = best_suggestion - grid_element / 2.
        pb = t

    while abs(pb - pa) > 1.0 / LARGE_NUMBER:
        x = [pb - (pb - pa) / gr, pa + (pb - pa) / gr]
        y = [goal_function(i_series, n_series, a, b, delta, linear, quadratic, strength_of_inference, v, o, c, x[m], importance_of_variance) for m in range(2)]
        if y[0] < y[1]:
            pb = x[1]
        else:
            pa = x[0]

    best_suggestion = (pa + pb) / 2.0
    goal_f[0] = best_suggestion
    goal_f[1] = goal_function(i_series, n_series, a, b, delta, linear, quadratic, strength_of_inference, v, o, c, best_suggestion, importance_of_variance)
    return best_suggestion


"""
/* 8 */
/***************************************************************************
point_solver() - guess the value of a given missing point.
***************************************************************************/
"""

def point_solver(
    n_series,          # number of time series
    n_days,            # number of days in history
    i_series,          # the point's time series
    i_date,            # the point's date
    raw_data,          # daily reported closing values
    adj_data,          # daily reported adjustment factors 
    val_score,         # output: degree of impact with respect to each TS
    delta,             # output: sensitivity of covariance
    linear,            # output: linear term of sensitivity of variance
    quadratic,         # output: quadratic term of sensitivity of variance
    goal_f,
    strength,
    my_variance,
    other_variance,
    covariance,
    cor_data           # variance / correlation vector (n_series)
):
    
    """
    Calculates the impact of removing a single data point on the correlation and variance of time series.

    Args:
        n_series: The number of time series.
        n_days: The number of days in the historical data.
        i_series: The index of the time series containing the point to be removed.
        i_date: The index of the date of the point to be removed.
        raw_data: A 1D array of raw closing values for all time series.
        adj_data: A 1D array of adjustment factors for all time series.
        val_score: Output array to store the impact of removing the point on each time series.
        delta: Output array to store the sensitivity of covariance.
        linear: Output array to store the linear term of sensitivity of variance.
        quadratic: Output array to store the quadratic term of sensitivity of variance.
        goal_f: Placeholder for future use.
        strength: Output array to store the estimation strength.
        my_variance: Output array to store the variance of the time series containing the point.
        other_variance: Output array to store the variance of other time series.
        covariance: Output array to store the covariance between time series.
        cor_data: Output array to store the correlation between time series.

    Returns:
        The current version of the library.
    """

    if n_days < 1:
        return ERROR_THE_LENGTH_OF_HISTORY_MUST_BE_POSITIVE
    if n_series < 1:
        return ERROR_THE_NUMBER_OF_TS_MUST_BE_POSITIVE

    # Allocate memory
    estimation_strength, cov  = 0.0, 0.0
    added_strength, added_cov = 0.0, 0.0
    
    projvar   = np.zeros(NTS)
    added_var = np.zeros(NTS)
    
    added_stats = [PairStats(0, 0, 0.0, [0.0, 0.0], 0.0, 0.0, 0.0) for _ in range(n_days)]
    stats       = [PairStats(0, 0, 0.0, [0.0, 0.0], 0.0, 0.0, 0.0) for _ in range(n_days)]
    added_stats = [PairStats(0, 0, 0.0, [0.0, 0.0], 0.0, 0.0, 0.0) for _ in range(n_days)]
    modified_ts = [TsPoint(0, 0, 0.0, 0.0) for _ in range(n_days)]
    f = [[TsPoint(0, 0, 0.0, 0.0) for _ in range(n_days)] for _ in range(NTS)]
    y = [[TsPoint(0, 0, 0.0, 0.0) for _ in range(n_days)] for _ in range(NTS)]
    z = [[TsPoint(0, 0, 0.0, 0.0) for _ in range(n_days)] for _ in range(n_series)]

    for j in range(n_series):
        val_score[j] = 0.0
        for m in range(n_days):
            r = z[j][m]
            r.val = raw_data[m * n_series + j]
            r.adj = adj_data[m * n_series + j]
        interval(z[j], 0, n_days - 1)

    r = z[i_series]
    r_palandri = f[0]
    q_palandri = f[PARTNER]

    for j in range(n_series):  # process the TS pair, including with itself
        q = z[j]
        alessandro_palandri(r, q, r_palandri, q_palandri, 0, n_days - 1, estimation_strength, cov, projvar, stats)
        if j == i_series:
            ts_involved = 1
            cor_data[j] = math.sqrt(projvar[0] / estimation_strength) if estimation_strength > 0.0 else 0.0
        else:
            ts_involved = NTS
            cor_data[j] = corr_value(cov, projvar)

        if r_palandri[i_date].val != MISSING_VALUE:  # find impact
            my_start = r_palandri[i_date].head
            if my_start < 0:
                my_start = r[i_date].head
            if my_start < 0:
                my_start = i_date
            if i_date < n_days - 1:
                my_end = r_palandri[i_date + 1].tail
                if my_end > n_days - 1:
                    my_end = r[i_date + 1].tail
                if my_end > n_days - 1:
                    my_end = i_date
            else:
                my_end = i_date
            if my_end > my_start:  # otherwise, no impact
                partner_start = q_palandri[my_start + 1].head
                if partner_start < 0:
                    partner_start = i_date
                partner_end = q_palandri[my_end].tail
                if partner_end > n_days - 1:
                    partner_end = i_date
                common_start = min(my_start, partner_start)
                common_end = max(my_end, partner_end)
                local_strength = local_cov = 0.0
                local_var = [0.0, 0.0]
                for d in range(common_start, common_end + 1):
                    modified_ts[d].val = r[d].val
                    modified_ts[d].adj = r[d].adj
                    if stats[d].left_point >= common_start and stats[d].rght_point <= common_end:
                        local_strength += stats[d].estimation_strength
                        local_cov += stats[d].covariance
                        local_var[0] += stats[d].variance[0]
                        local_var[1] += stats[d].variance[PARTNER]
                xtemp = modified_ts[i_date].val
                modified_ts[i_date].val = MISSING_VALUE  # remove the point
                interval(modified_ts, common_start, common_end)
                alessandro_palandri(modified_ts, q, y[0], y[1], common_start, common_end, added_strength, added_cov, added_var, added_stats)
                new_cov = cov - local_cov + added_cov
                new_var = [projvar[d] - local_var[d] + added_var[d] for d in range(NTS)]
                corr_without = corr_value(new_cov, new_var)
                if ts_involved == 1:  # the impact = reduction in volatility
                    variance_without = new_var[0] * (estimation_strength / (estimation_strength - 1.0) if estimation_strength > 1.0 else 1.0)
                    if projvar[0] > variance_without:  # impact > 0
                        impact = 1.0 - math.sqrt(variance_without / projvar[0])
                    elif projvar[0] < variance_without:  # impact < 0
                        impact = math.sqrt(projvar[0] / variance_without) - 1.0
                    else:
                        impact = 0.0
                else:  # the measure is average improvement in squared correlation, in percentage points
                    impact = corr_without - cor_data[j]
                modified_ts[i_date].val = xtemp  # restore the value
                val_score[j] = impact
        b = stats[i_date]
        strength[j] = estimation_strength
        my_variance[j] = projvar[0]
        other_variance[j] = projvar[1]
        linear[j] = b.linear
        quadratic[j] = b.quadratic
        covariance[j] = cov
        if ts_involved != 1:
            delta[j] = b.delta

    min_val = max_val = 0.0
    for m in range(n_days):
        temp_val = raw_data[m * n_series + i_series]
        if temp_val == MISSING_VALUE:
            continue
        if min_val > temp_val:
            min_val = temp_val
        if max_val < temp_val:
            max_val = temp_val

    golden_ratio_search(i_series, n_series, cor_data, val_score, delta, linear, quadratic, strength, my_variance, other_variance, covariance, goal_f, min_val - 1.0, max_val + 1.0)
    

    return __version__

