"""Plotly figürleri. Hesaplama `analysis` modülünde yapılır; burada sadece çizilir."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

from optistudy.analysis import TOTAL_LABEL, BacktestResult, NetForecast, fit_trend, net_series


@dataclass(frozen=True)
class ChartColors:
    """Bir görünüm modu (açık/koyu) için grafik renkleri.

    Seri ve tahmin renkleri, renk körlüğü ayrımı ve zemin kontrastı için doğrulanmış
    paletten seçildi. Açık ve koyu zemin için ayrı tonlar gerekir: açık moddaki turuncu
    koyu zeminde fazla parlak kalıyor.
    """

    series: str     # deneme netleri
    forecast: str   # tahmin noktası/aralığı; backtest'te aralık dışı kalan net
    trend: str      # regresyon doğrusu (ikincil, nötr gri)


LIGHT_COLORS = ChartColors(series="#2a78d6", forecast="#eb6834", trend="#898781")
DARK_COLORS = ChartColors(series="#3987e5", forecast="#d95926", trend="#898781")


def chart_colors(dark: bool) -> ChartColors:
    """Görünüm moduna göre renk seti."""
    return DARK_COLORS if dark else LIGHT_COLORS


# Arayüzde kullanılacak plotly ayarı: araç çubuğu gizli
PLOTLY_CONFIG: dict[str, bool] = {"displayModeBar": False}

_LEGEND = {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right", "title": None}


def _static_axes(fig: go.Figure) -> go.Figure:
    """Sürükleyerek yakınlaştırmayı kapatır; üzerine gelince bilgi gösterme açık kalır."""
    fig.update_layout(dragmode=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def net_trend_figure(
    results: pd.DataFrame, subject: str, colors: ChartColors = LIGHT_COLORS,
) -> go.Figure | None:
    """Deneme netleri ve (en az 3 deneme varsa) doğrusal trend çizgisi.

    Hiç deneme yoksa `None` döner.
    """
    series = net_series(results, subject)
    if series.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["date"], y=series["net"],
        mode="lines+markers", name="Net",
        line={"color": colors.series, "width": 2},
        marker={"color": colors.series, "size": 8},
        customdata=series[["publisher"]],
        hovertemplate="%{x|%d.%m.%Y}<br>%{customdata[0]}<br>Net: %{y:.2f}<extra></extra>",
    ))

    trend = fit_trend(series)
    if trend is not None:
        ends = pd.Series(pd.to_datetime([series["date"].min(), series["date"].max()]))
        fig.add_trace(go.Scatter(
            x=ends, y=trend.fitted(ends),
            mode="lines", name="Trend",
            line={"color": colors.trend, "width": 2, "dash": "dash"},
            hovertemplate=f"Trend: {trend.slope_per_week:+.2f} net/hafta<extra></extra>",
        ))

    fig.update_layout(
        title=f"{subject} neti" if subject != TOTAL_LABEL else "Toplam net",
        xaxis_title=None, yaxis_title="Net",
        hovermode="closest", height=380,
        legend=_LEGEND,
        margin={"t": 60, "b": 10},
    )
    fig.update_xaxes(tickformat="%d.%m", showgrid=False)
    return _static_axes(fig)


def consistency_figure(summary: pd.DataFrame, colors: ChartColors = LIGHT_COLORS) -> go.Figure | None:
    """Ders bazlı tutarlılık (son 5 denemenin standart sapması) çubuk grafiği.

    Toplam satırı farklı ölçekte olduğu için dahil edilmez. Hiçbir ders için
    değer yoksa `None` döner.
    """
    data = summary[(summary["subject"] != TOTAL_LABEL) & summary["std_last5"].notna()]
    if data.empty:
        return None

    fig = go.Figure(go.Bar(
        x=data["subject"], y=data["std_last5"],
        marker={"color": colors.series, "cornerradius": 4},
        text=data["std_last5"].round(2), textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>Standart sapma: %{y:.2f} net<extra></extra>",
    ))
    fig.update_layout(
        title="Tutarlılık (son 5 deneme, standart sapma)",
        yaxis_title="Standart sapma (net)", xaxis_title=None,
        showlegend=False, height=320, bargap=0.55, margin={"t": 60, "b": 10},
    )
    return _static_axes(fig)


def forecast_figure(
    results: pd.DataFrame, subject: str, forecasts: list[NetForecast],
    colors: ChartColors = LIGHT_COLORS, target_net: float | None = None,
) -> go.Figure | None:
    """Geçmiş netler, trend çizgisi ve her tahmin için %80 tahmin aralığı (hata çubuğu).

    Trend çizgisi en uzak tahmin tarihine kadar uzatılır. `target_net` verilirse hedef
    yatay noktalı çizgiyle gösterilir. Hiç deneme yoksa `None` döner.
    """
    fig = net_trend_figure(results, subject, colors)
    if fig is not None and target_net is not None:
        fig.add_hline(
            y=target_net, line={"color": colors.trend, "width": 1.5, "dash": "dot"},
            annotation_text=f"Hedef {target_net:g}", annotation_position="top left",
        )
    if fig is None or not forecasts:
        return fig

    series = net_series(results, subject)
    trend = fit_trend(series)
    if trend is not None:
        last = max(f.target_date for f in forecasts)
        ends = pd.Series(pd.to_datetime([series["date"].min(), last]))
        fig.data[1].update(x=ends, y=trend.fitted(ends))

    fig.add_trace(go.Scatter(
        x=[f.target_date for f in forecasts], y=[f.point for f in forecasts],
        mode="markers", name=f"Tahmin (%{forecasts[0].level * 100:.0f} aralık)",
        marker={"color": colors.forecast, "size": 10, "symbol": "diamond"},
        error_y={
            "type": "data", "symmetric": False, "color": colors.forecast, "thickness": 2, "width": 6,
            "array": [f.high - f.point for f in forecasts],
            "arrayminus": [f.point - f.low for f in forecasts],
        },
        customdata=[[f.low, f.high] for f in forecasts],
        hovertemplate="%{x|%d.%m.%Y}<br>Tahmin: %{y:.1f} net<br>"
                      "Aralık: %{customdata[0]:.1f} – %{customdata[1]:.1f}<extra></extra>",
    ))
    return fig


def backtest_figure(result: BacktestResult, colors: ChartColors = LIGHT_COLORS) -> go.Figure:
    """Backtest: her adımda tahmin aralığı ve gerçekleşen net.

    Aralık içinde kalanlar daire, dışında kalanlar çarpı işaretiyle (renk tek başına bilgi taşımaz).
    """
    steps = result.steps
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps["date"], y=steps["point"], mode="markers",
        name=f"Tahmin (%{result.level * 100:.0f} aralık)",
        marker={"color": colors.trend, "size": 8, "symbol": "line-ew-open"},
        error_y={
            "type": "data", "symmetric": False, "color": colors.trend, "thickness": 2, "width": 6,
            "array": steps["high"] - steps["point"], "arrayminus": steps["point"] - steps["low"],
        },
        customdata=steps[["low", "high"]],
        hovertemplate="%{x|%d.%m.%Y}<br>Tahmin: %{y:.1f}<br>"
                      "Aralık: %{customdata[0]:.1f} – %{customdata[1]:.1f}<extra></extra>",
    ))
    for hit, name, color, symbol in (
        (True, "Gerçek net (aralıkta)", colors.series, "circle"),
        (False, "Gerçek net (aralık dışı)", colors.forecast, "x"),
    ):
        part = steps[steps["hit"] == hit]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["date"], y=part["actual"], mode="markers", name=name,
            marker={"color": color, "size": 10, "symbol": symbol},
            hovertemplate="%{x|%d.%m.%Y}<br>Gerçek: %{y:.2f} net<extra></extra>",
        ))
    fig.update_layout(
        title="Geçmiş tahminler tuttu mu?",
        yaxis_title="Net", xaxis_title=None, height=340,
        legend=_LEGEND,
        margin={"t": 60, "b": 10},
    )
    fig.update_xaxes(tickformat="%d.%m", showgrid=False)
    return _static_axes(fig)


def priority_figure(focus: pd.DataFrame, colors: ChartColors = LIGHT_COLORS) -> go.Figure | None:
    """Haftalık öncelik listesi: konu başına sınavda beklenen kayıp soru (yatay çubuk).

    En öncelikli konu en üstte. Liste boşsa `None` döner.
    """
    if focus.empty:
        return None
    data = focus.iloc[::-1]
    labels = data["subject"] + " · " + data["topic"]
    fig = go.Figure(go.Bar(
        x=data["priority"], y=labels, orientation="h",
        marker={"color": colors.series, "cornerradius": 4},
        text=data["priority"].round(1), textposition="outside", cliponaxis=False,
        customdata=data[["weight", "adjusted_error_rate"]],
        hovertemplate="%{y}<br>Beklenen kayıp: %{x:.2f} soru/sınav<br>"
                      "Soru ağırlığı: %{customdata[0]:.1f} · Düzeltilmiş hata oranı: "
                      "%{customdata[1]:.0%}<extra></extra>",
    ))
    fig.update_layout(
        title="Bu hafta odaklan",
        xaxis_title="Sınav başına beklenen kayıp soru", yaxis_title=None,
        showlegend=False, height=110 + 48 * len(data), bargap=0.4,
        margin={"t": 60, "b": 10, "l": 10, "r": 30},
    )
    fig.update_yaxes(showgrid=False)
    return _static_axes(fig)
