"""Utility functions"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_selected(df, columns, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    df_plt = df.loc[start_index:end_index, columns]
    df_plt = normalize_data(df_plt)
    plot_data(df_plt)


def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return Path(base_dir).joinpath("{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date",
                              parse_dates=True, usecols=['Date', 'Adj Close'],
                              na_values=['nan'])

        df_temp = df_temp.rename(columns={'Adj Close': symbol})

        how = 'inner' if symbol is 'SPY' else 'left'
        df = df.join(df_temp, how=how)

    return df


def normalize_data(df):
    return df / df.max()


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling(window=window).mean()


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return values.rolling(window=window).std()


def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm + 2 * rstd
    lower_band = rm - 2 * rstd
    return upper_band, lower_band


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.iloc[0] = 0  # set daily returns for row 0 to 0
    return daily_returns


def test_run():
    # Get Data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'XOM', 'GLD']
    df = get_data(symbols, dates)

    # Slice and plot
    # plot_selected(df, ['SPY', 'XOM', 'GLD'], '2010-03-01', '2010-04-01')

    ## Compute Bollinger Bands ##
    # # 1. Compute rolling mea
    # rm_SPY = get_rolling_mean(df['SPY'], window=20)
    #
    # # 2. Compute rolling standard deviation
    # rstd_SPY = get_rolling_std(df['SPY'], window=20)
    #
    # # 3. Compute upper and lower bands
    # upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)
    #
    # # Plot raw SPY values, rolling mean and Bollinger Bands
    # ax = df['SPY'].plot(title='SPY rolling mean', label='SPY')
    # rm_SPY.plot(label='Rolling mean', ax=ax)
    # upper_band.plot(label='upper band', ax=ax)
    # lower_band.plot(label='lower band', ax=ax)
    #
    # # Add axis labels and legend
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Price")
    # ax.legend(loc='upper left')
    # plt.show()

    ## Compute daily returns ##
    daily_returns = compute_daily_returns(df)
    # plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")
    cumulative_returns = daily_returns.cumsum()
    # plot_data(cumulative_returns, title="Cumulative returns", ylabel="Cumulative Sum")

    ## Plot a histogram ##
    # daily_returns['SPY'].hist(bins=20, label="SPY")
    # daily_returns['XOM'].hist(bins=20, label="XOM")
    # plt.legend(loc='upper right')
    #
    # # Get mean and standard deviation
    # mean = daily_returns['SPY'].mean()
    # print(f"mean={mean}")
    # std = daily_returns['SPY'].std()
    # print(f"std={std}")
    #
    # plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    # plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
    # plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    # plt.show()
    #
    # # Compute kurtosis
    # print(f"kurtosis={daily_returns['SPY'].kurtosis()}")

    ## Plot a Scatterplot ##
    # Scatterplot SPY vs XOm
    daily_returns.plot(kind='scatter', x='SPY', y='XOM')
    beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'], 1)
    plt.plot(daily_returns['SPY'], beta_XOM*daily_returns['SPY'] + alpha_XOM, '-', color='r')
    plt.show()

    # Scatterplot SPY vs GLD
    daily_returns.plot(kind='scatter', x='SPY', y='GLD')
    plt.show()

    # Calculate correlation coefficient
    print(daily_returns.corr(method='pearson'))

if __name__ == "__main__":
    test_run()
