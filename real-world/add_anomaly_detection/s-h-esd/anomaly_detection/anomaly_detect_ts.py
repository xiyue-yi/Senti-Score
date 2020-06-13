"""
Description:

     A technique for detecting anomalies in seasonal univariate time
     series where the input is a series of <timestamp, count> pairs.


Usage:

     anomaly_detect_ts(x, max_anoms=0.1, direction="pos", alpha=0.05, only_last=None,
                      threshold="None", e_value=False, longterm=False, piecewise_median_period_weeks=2,
                      plot=False, y_log=False, xlabel="", ylabel="count", title=None, verbose=False)

Arguments:

       x: Time series as a two column data frame where the first column
          consists of the timestamps and the second column consists of
          the observations.

max_anoms: Maximum number of anomalies that S-H-ESD will detect as a
          percentage of the data.

direction: Directionality of the anomalies to be detected. Options are:
          "pos" | "neg" | "both".

   alpha: The level of statistical significance with which to accept or
          reject anomalies.

only_last: Find and report anomalies only within the last day or hr in
          the time series. None | "day" | "hr".

threshold: Only report positive going anoms above the threshold
          specified. Options are: None | "med_max" | "p95" |
          "p99".

 e_value: Add an additional column to the anoms output containing the
          expected value.

longterm: Increase anom detection efficacy for time series that are
         greater than a month. See Details below.

piecewise_median_period_weeks: The piecewise median time window as
          described in Vallis, Hochenbaum, and Kejariwal (2014).
          Defaults to 2.

    plot: A flag indicating if a plot with both the time series and the
          estimated anoms, indicated by circles, should also be
          returned.

   y_log: Apply log scaling to the y-axis. This helps with viewing
          plots that have extremely large positive anomalies relative
          to the rest of the data.

  xlabel: X-axis label to be added to the output plot.

  ylabel: Y-axis label to be added to the output plot.

   title: Title for the output plot.

 verbose: Enable debug messages
 
 resampling: whether ms or sec granularity should be resampled to min granularity. 
             Defaults to False.
             
 period_override: Override the auto-generated period
                  Defaults to None

Details:

     "longterm" This option should be set when the input time series
     is longer than a month. The option enables the approach described
     in Vallis, Hochenbaum, and Kejariwal (2014).
     "threshold" Filter all negative anomalies and those anomalies
     whose magnitude is smaller than one of the specified thresholds
     which include: the median of the daily max values (med_max), the
     95th percentile of the daily max values (p95), and the 99th
     percentile of the daily max values (p99).

Value:

    The returned value is a list with the following components.

    anoms: Data frame containing timestamps, values, and optionally
          expected values.

    plot: A graphical object if plotting was requested by the user. The
          plot contains the estimated anomalies annotated on the input
          time series.
     "threshold" Filter all negative anomalies and those anomalies
     whose magnitude is smaller than one of the specified thresholds
     which include: the median of the daily max values (med_max), the
     95th percentile of the daily max values (p95), and the 99th
     percentile of the daily max values (p99).

Value:

     The returned value is a list with the following components.

     anoms: Data frame containing timestamps, values, and optionally
          expected values.

     plot: A graphical object if plotting was requested by the user. The
          plot contains the estimated anomalies annotated on the input
          time series.
     One can save "anoms" to a file in the following fashion:
     write.csv(<return list name>[["anoms"]], file=<filename>)

     One can save "plot" to a file in the following fashion:
     ggsave(<filename>, plot=<return list name>[["plot"]])

References:

     Vallis, O., Hochenbaum, J. and Kejariwal, A., (2014) "A Novel
     Technique for Long-Term Anomaly Detection in the Cloud", 6th
     USENIX, Philadelphia, PA.

     Rosner, B., (May 1983), "Percentage Points for a Generalized ESD
     Many-Outlier Procedure" , Technometrics, 25(2), pp. 165-172.

See Also:

     anomaly_detect_vec

Examples:
     # To detect all anomalies
     anomaly_detect_ts(raw_data, max_anoms=0.02, direction="both", plot=True)
     # To detect only the anomalies in the last day, run the following:
     anomaly_detect_ts(raw_data, max_anoms=0.02, direction="both", only_last="day", plot=True)
     # To detect only the anomalies in the last hr, run the following:
     anomaly_detect_ts(raw_data, max_anoms=0.02, direction="both", only_last="hr", plot=True)
     # To detect only the anomalies in the last hr and resample data of ms or sec granularity:
     anomaly_detect_ts(raw_data, max_anoms=0.02, direction="both", only_last="hr", plot=True, resampling=True)
     # To detect anomalies in the last day specifying a period of 1440
     anomaly_detect_ts(raw_data, max_anoms=0.02, direction="both", only_last="hr", period_override=1440)

"""

import numpy as np
import scipy as sp
import pandas as pd
import datetime
import statsmodels.api as sm
import logging


logger = logging.getLogger(__name__)


def _handle_granularity_error(level):
    """
    Raises ValueError with detailed error message if one of the two situations is true:
      1. calculated granularity is less than minute (sec or ms)
      2. resampling is not enabled for situations where calculated granularity < min

      level : String
        the granularity that is below the min threshold
    """
    e_message = '%s granularity is not supported. Ensure granularity => minute or enable resampling' % level
    raise ValueError(e_message)


def _resample_to_min(data, period_override=None):
    """
    Resamples a data set to the min level of granularity

      data : pandas DataFrame
        input Pandas DataFrame
      period_override : int
        indicates whether resampling should be done with overridden value instead of min (1440)

    """
    data = data.resample('60s', label='right').sum()
    if _override_period(period_override):
        period = period_override
    else:
        period = 1440
    return (data, period)


def _override_period(period_override):
    """
    Indicates whether period can be overridden if the period derived from granularity does
    not match the generated period.

      period_override : int
        the user-specified period that overrides the value calculated from granularity
    """
    return period_override is not None


def _get_period(gran_period, period_arg=None):
    """
    Returns the generated period or overridden period depending upon the period_arg
      gran_period : int
        the period generated from the granularity
      period_arg : the period override value that is either None or an int
        the period to override the period generated from granularity
    """
    if _override_period(period_arg):
        return period_arg
    else:
        return gran_period


def _get_data_tuple(raw_data, period_override, resampling=False):
    """
    Generates a tuple consisting of processed input data, a calculated or overridden period, and granularity

      raw_data : pandas DataFrame
        input data
      period_override : int
        period specified in the anomaly_detect_ts parameter list, None if it is not provided
      resampling : True | False
        indicates whether the raw_data should be resampled to a supporting granularity, if applicable
    """
    data = raw_data.sort_index()
    timediff = _get_time_diff(data)
    if timediff.days > 0:
        period = _get_period(7, period_override)
        granularity = 'day'
    elif timediff.seconds / 60 / 60 >= 1:
        granularity = 'hr'
        period = _get_period(24, period_override)
    elif timediff.seconds / 60 >= 1:
        granularity = 'min'
        period = _get_period(1440, period_override)
    elif timediff.seconds > 0:
        granularity = 'sec'
    elif timediff.seconds > 0:
        granularity = 'sec'

        '''
           Aggregate data to minute level of granularity if data stream granularity is sec and
           resampling=True. If resampling=False, raise ValueError
        '''
        if resampling is True:
            period = _resample_to_min(data, period_override)
        else:
            _handle_granularity_error('sec')
    else:
        '''
           Aggregate data to minute level of granularity if data stream granularity is ms and
           resampling=True. If resampling=False, raise ValueError
        '''
        if resampling is True:
            data, period = _resample_to_min(data, period_override)
            granularity = None
        else:
            _handle_granularity_error('ms')

    return (data, period, granularity)


def _get_time_diff(data):
    """
    Generates the time difference used to determine granularity and
    to generate the period

      data : pandas DataFrame
        composed of input data
    """
    return data.index[1] - data.index[0]


def _get_max_anoms(data, max_anoms):
    """
    Returns the max_anoms parameter used for S-H-ESD time series anomaly detection

      data : pandas DataFrame
        composed of input data
      max_anoms : float
        the input max_anoms
    """
    if max_anoms == 0:
        logger.warning('0 max_anoms results in max_outliers being 0.')
    return 1 / data.size if max_anoms < 1 / data.size else max_anoms


def _process_long_term_data(data, period, granularity, piecewise_median_period_weeks):
    """
    Processes result set when longterm is set to true

      data : list of floats
        the result set of anoms
      period : int
        the calculated or overridden period value
      granularity : string
        the calculated or overridden granularity
      piecewise_median_period_weeks : int
        used to determine days and observations per period
    """
    # Pre-allocate list with size equal to the number of piecewise_median_period_weeks chunks in x + any left over chunk
    # handle edge cases for daily and single column data period lengths
    num_obs_in_period = period * piecewise_median_period_weeks + \
        1 if granularity == 'day' else period * 7 * piecewise_median_period_weeks
    num_days_in_period = (7 * piecewise_median_period_weeks) + \
        1 if granularity == 'day' else (7 * piecewise_median_period_weeks)

    all_data = []
    # Subset x into piecewise_median_period_weeks chunks
    for i in range(1, data.size + 1, num_obs_in_period):
        start_date = data.index[i]
        # if there is at least 14 days left, subset it, otherwise subset last_date - 14 days
        end_date = start_date + datetime.timedelta(days=num_days_in_period)
        if end_date < data.index[-1]:
            all_data.append(
                data.loc[lambda x: (x.index >= start_date) & (x.index <= end_date)])
        else:
            all_data.append(
                data.loc[lambda x: x.index >= data.index[-1] - datetime.timedelta(days=num_days_in_period)])
    return all_data


def _get_only_last_results(data, all_anoms, granularity, only_last):
    """
    Returns the results from the last day or hour only

      data : pandas DataFrame
        input data set
      all_anoms : list of floats
        all of the anomalies returned by the algorithm
      granularity : string day | hr | min
        The supported granularity value
      only_last : string day | hr
        The subset of anomalies to be returned
    """
    start_date = data.index[-1] - datetime.timedelta(days=7)
    start_anoms = data.index[-1] - datetime.timedelta(days=1)

    if only_last == 'hr':
        # We need to change start_date and start_anoms for the hourly only_last option
        start_date = datetime.datetime.combine(
            (data.index[-1] - datetime.timedelta(days=2)).date(), datetime.time.min)
        start_anoms = data.index[-1] - datetime.timedelta(hours=1)

    # subset the last days worth of data
    x_subset_single_day = data.loc[data.index > start_anoms]
    # When plotting anoms for the last day only we only show the previous weeks data
    x_subset_week = data.loc[lambda df: (
        df.index <= start_anoms) & (df.index > start_date)]
    return all_anoms.loc[all_anoms.index >= x_subset_single_day.index[0]]


def _get_plot_breaks(granularity, only_last):
    """
    Generates the breaks used in plotting

      granularity : string
        the supported granularity value
      only_last : True | False
        indicates whether only the last day or hour is returned and to be plotted
    """
    if granularity == 'day':
        breaks = 3 * 12
    elif only_last == 'day':
        breaks = 12
    else:
        breaks = 3
    return breaks


def _perform_threshold_filter(anoms, periodic_max, threshold):
    """
    Filters the list of anomalies per the threshold filter

      anoms : list of floats
        the anoms returned by the algorithm
      periodic_max : float
        calculated daily max value
      threshold : med_max" | "p95" | "p99"
        user-specified threshold value used to filter anoms
    """
    if threshold == 'med_max':
        thresh = periodic_max.median()
    elif threshold == 'p95':
        thresh = periodic_max.quantile(0.95)
    elif threshold == 'p99':
        thresh = periodic_max.quantile(0.99)
    else:
        raise AttributeError(
            'Invalid threshold, threshold options are None | med_max | p95 | p99')

    return anoms.loc[anoms.values >= thresh]


def _get_max_outliers(data, max_percent_anomalies):
    """
    Calculates the max_outliers for an input data set

      data : pandas DataFrame
        the input data set
      max_percent_anomalies : float
        the input maximum number of anomalies per percent of data set values
    """
    max_outliers = int(np.trunc(data.size * max_percent_anomalies))
    assert max_outliers, 'With longterm=True, AnomalyDetection splits the data into 2 week periods by default. You have {0} observations in a period, which is too few. Set a higher piecewise_median_period_weeks.'.format(
        data.size)
    return max_outliers


def _get_decomposed_data_tuple(data, num_obs_per_period):
    """
    Returns a tuple consisting of two versions of the input data set: seasonally-decomposed and smoothed

      data : pandas DataFrame
        the input data set
      num_obs_per_period : int
        the number of observations in each period
    """
    print(num_obs_per_period,'frequency')
    decomposed = sm.tsa.seasonal_decompose(
        data, freq=num_obs_per_period, two_sided=False)
    #print(decomposed.seasonal)
    smoothed = data - decomposed.resid.fillna(0)
    mean_data = data.mean()
    data = data - decomposed.seasonal - data.mean()
    s_data = data + mean_data
    return (data, s_data, smoothed)


def anomaly_detect_ts(x, max_anoms=0.1, direction="pos", alpha=0.05, only_last=None,
                      threshold=None, e_value=False, longterm=False, piecewise_median_period_weeks=2,
                      plot=False, y_log=False, xlabel="", ylabel="count", title='shesd output: ', verbose=False,
                      dropna=False, resampling=False, period_override=None):

    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("The debug logs will be logged because verbose=%s", verbose)
    print(x.index.dtype)
    x.index = x.index.astype('datetime64[ns]')
    print(x.index.dtype)
    # validation
    assert isinstance(x, pd.Series), 'Data must be a series(Pandas.Series)'
    assert x.values.dtype in [int, float], 'Values of the series must be number'
    assert x.index.dtype == np.dtype('datetime64[ns]'), 'Index of the series must be datetime'
    assert max_anoms <= 0.49 and max_anoms >= 0, 'max_anoms must be non-negative and less than 50% '
    assert direction in ['pos', 'neg', 'both'], 'direction options: pos | neg | both'
    assert only_last in [None, 'day', 'hr'], 'only_last options: None | day | hr'
    assert threshold in [None, 'med_max', 'p95', 'p99'], 'threshold options: None | med_max | p95 | p99'
    assert piecewise_median_period_weeks >= 2, 'piecewise_median_period_weeks must be greater than 2 weeks'
    logger.debug('Completed validation of input parameters')

    if alpha < 0.01 or alpha > 0.1:
        logger.warning('alpha is the statistical significance and is usually between 0.01 and 0.1')

    data, period, granularity = _get_data_tuple(x, period_override, resampling)
    print(data,period,granularity)
    if granularity is 'day':
        num_days_per_line = 7
        only_last = 'day' if only_last == 'hr' else only_last

    max_anoms = _get_max_anoms(data, max_anoms)

    # If longterm is enabled, break the data into subset data frames and store in all_data
    all_data = _process_long_term_data(data, period, granularity, piecewise_median_period_weeks) if longterm else [data] 
    all_anoms = pd.Series()
    seasonal_plus_trend = pd.Series()

    # Detect anomalies on all data (either entire data in one-pass, or in 2 week blocks if longterm=True)
    for series in all_data:
        shesd = _detect_anoms(series, k=max_anoms, alpha=alpha, num_obs_per_period=period, use_decomp=True,
                              use_esd=False, direction=direction, verbose=verbose)
        shesd_anoms = shesd['anoms']
        shesd_stl = shesd['stl']
        shesd_data = shesd['data']
        s_data = shesd['s_data']
        degree = shesd['degree']
        de_dis = shesd['de_dis']
        print('nanother data')
        print(shesd_data)

        # -- Step 3: Use detected anomaly timestamps to extract the actual anomalies (timestamp and value) from the data
        anoms = pd.Series() if shesd_anoms.empty else series.loc[shesd_anoms.index]

        # Filter the anomalies using one of the thresholding functions if applicable
        if threshold:
            # Calculate daily max values
            periodic_max = data.resample('1D').max()
            anoms = _perform_threshold_filter(anoms, periodic_max, threshold)

        all_anoms = all_anoms.append(anoms)
        seasonal_plus_trend = seasonal_plus_trend.append(shesd_stl)

    # De-dupe
    all_anoms.drop_duplicates(inplace=True)
    seasonal_plus_trend.drop_duplicates(inplace=True)

    # If only_last is specified, create a subset of the data corresponding to the most recent day or hour
    if only_last:
        all_anoms = _get_only_last_results(
            data, all_anoms, granularity, only_last)

    # If there are no anoms, log it and return an empty anoms result
    if all_anoms.empty:
        if verbose:
            logger.info('No anomalies detected.')

        return {
            'anoms': pd.Series(),
            'plot': None
        }

    if plot:
        # TODO additional refactoring and logic needed to support plotting
        num_days_per_line
        #breaks = _get_plot_breaks(granularity, only_last)
        # x_subset_week
        raise Exception('TODO: Unsupported now')

    return {
        'anoms': all_anoms,
        'expected': seasonal_plus_trend if e_value else None,
        'plot': 'TODO' if plot else None,
        'shesd_data': shesd_data,
        's_data': s_data,
        'degree': degree,
        'de_dis': de_dis,
        'free': shesd['free']
    }


def _detect_anoms(data, k=0.49, alpha=0.05, num_obs_per_period=4,
                  use_decomp=True, use_esd=False, direction="pos", verbose=False):
    """
    Detects anomalies in a time series using S-H-ESD.

    Args:
         data: Time series to perform anomaly detection on.
         k: Maximum number of anomalies that S-H-ESD will detect as a percentage of the data.
         alpha: The level of statistical significance with which to accept or reject anomalies.
         num_obs_per_period: Defines the number of observations in a single period, and used during seasonal decomposition.
         use_decomp: Use seasonal decomposition during anomaly detection.
         use_esd: Uses regular ESD instead of hybrid-ESD. Note hybrid-ESD is more statistically robust.
         one_tail: If TRUE only positive or negative going anomalies are detected depending on if upper_tail is TRUE or FALSE.
         upper_tail: If TRUE and one_tail is also TRUE, detect only positive going (right-tailed) anomalies. If FALSE and one_tail is TRUE, only detect negative (left-tailed) anomalies.
         verbose: Additionally printing for debugging.
    Returns:
       A list containing the anomalies (anoms) and decomposition components (stl).
    """

    # validation
    assert num_obs_per_period, "must supply period length for time series decomposition"
    assert direction in ['pos', 'neg',
                         'both'], 'direction options: pos | neg | both'
    assert data.size >= num_obs_per_period * \
        2, 'Anomaly detection needs at least 2 periods worth of data'
    assert data[data.isnull(
    )].empty, 'Data contains NA. We suggest replacing NA with interpolated values before detecting anomaly'

    # conversion
    one_tail = True if direction in ['pos', 'neg'] else False
    upper_tail = True if direction in ['pos', 'both'] else False

    # -- Step 1: Decompose data. This returns a univariate remainder which will be used for anomaly detection. Optionally, we might NOT decompose.
    # Note: R use stl, but here we will use MA, the result may be different TODO.. Here need improvement
    #decomposed = sm.tsa.seasonal_decompose(data, freq=num_obs_per_period, two_sided=False)
    #smoothed = data - decomposed.resid.fillna(0)
    #data = data - decomposed.seasonal - data.mean()

    print(num_obs_per_period,'??')
    data, s_data, smoothed = _get_decomposed_data_tuple(data, num_obs_per_period)

    keep_data = data.copy()
    print('aanother data')
    print(keep_data)

    max_outliers = _get_max_outliers(data, k)

    R_idx = pd.Series()
    # 异常点 异常的程度 t/c
    degree_idx = pd.Series()
    # 异常备选点 异常的程度
    free_idx = pd.Series()
   
    n = data.size

    # Compute test statistic until r=max_outliers values have been
    # removed from the sample.
    for i in range(1, max_outliers + 1):
        if verbose:
            logger.info(i, '/', max_outliers, ' completed')

        if not data.mad():
            break
        
        if not one_tail:
            ares = abs(data - data.median())
        elif upper_tail:
            ares = data - data.median()
        else:
            ares = data.median() - data

        ares = ares / data.mad()
        '''
        if not one_tail:
            ares = abs(data - data.mean())
        elif upper_tail:
            ares = data - data.mean()
        else:
            ares = data.mean() - data

        ares = ares / data.std()
        '''

        print(ares.values == ares.max())
        tmp_anom_index = ares[ares.values == ares.max()].index
        cand = pd.Series(data.loc[tmp_anom_index], index=tmp_anom_index)

        data.drop(tmp_anom_index, inplace=True)

        # Compute critical value.
        p = 1 - alpha / (n - i + 1) if one_tail else (1 -
                                                      alpha / (2 * (n - i + 1)))
        t = sp.stats.t.ppf(p, n - i - 1)
        lam = t * (n - i) / np.sqrt((n - i - 1 + t ** 2) * (n - i + 1))
        
        degree = pd.Series(ares.max()/lam,index=tmp_anom_index)
        free_idx = free_idx.append(degree)
        if ares.max() > lam:
            R_idx = R_idx.append(cand)
        degree_idx = degree_idx.append(degree)
    
    print('nanother data')
    
    all_anomaly_indexes = get_all_anomaly_indexes(R_idx,keep_data)
    distance_list = []
    #for i,anomaly_index in enumerate(R_idx.index.values):
    for i,anomaly_index in enumerate(degree_idx.index.values):
        distance_list.append(get_distance(anomaly_index,keep_data,num_obs_per_period,all_anomaly_indexes))
        
    de_dis_idx = pd.Series()
    for i,anomaly_index in enumerate(degree_idx.index):
        tmp_key,tmp_index = [],[]
        tmp_key.append(degree_idx[anomaly_index]/distance_list[i])
        tmp_index.append(anomaly_index)
        de_dis = pd.Series(tmp_key,index=tmp_index)
        de_dis_idx = de_dis_idx.append(de_dis)    
    
    print(all_anomaly_indexes)
    return {
        'anoms': R_idx,
        'stl': smoothed,
        'data': keep_data,
        's_data': s_data,
        'degree': degree_idx,
        'de_dis': de_dis_idx,
        'free': free_idx
    }

def get_raw_index(series_index,  data):
    for i,row in enumerate(data.index.values):
        if (row == series_index):
            return i
def get_all_anomaly_indexes(anomaly_series,data):
    all_anomaly_indexes = []
    for i,anomaly_index in enumerate(anomaly_series.index.values):
        all_anomaly_indexes.append(get_raw_index(anomaly_index,  data))
    return all_anomaly_indexes
def get_distance(series_index,  data, window_size,all_anomaly_indexes):
    raw_index = get_raw_index(series_index,  data)
    quotient = int(raw_index/window_size)
    full_window_indexes = [quotient*window_size + i for i in range(window_size)]
    window_indexes = []
    flag = 0
    distance = 0.0
    for index in full_window_indexes:
        if index == raw_index:
            continue
        elif index not in all_anomaly_indexes:
            flag = 1
            distance += abs(index-raw_index)/window_size
    distance += float(1/window_size)
    distance *= 1/(window_size-2)
    return distance
