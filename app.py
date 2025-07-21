### TO RUN THIS APP TYPE: "streamlit run app.py" in terminal
### NEED TO BE IN THE SAME DIRECTORY AS THIS FILE


import streamlit as st
import pandas as pd
import numpy as np
import PyCO2SYS as pyco2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import math

st.set_page_config(page_title="Data Processing App", layout="wide")

st.title("📊 ISFET QC Processing App")
st.markdown("Upload your data, calculate in situ k0 from tris/bottles, visualize results, export QCed data.")
st.markdown("Test")

# Upload tab
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "Initial plots", "Calc k0","QCed data"])

with tab1:
    st.header("Upload Sensor File")
    data_file = st.file_uploader("CSV file with headers: DTUTC, VINT, TEMPC", type="csv")
    # data_file = "/Users/taylorwirth/Desktop/ISFET_QC_GUI/sensor_data_example_stable.csv"
    
    if data_file is not None:
        sen_df = pd.read_csv(data_file)
        sen_df['DTUTC'] = pd.to_datetime(sen_df['DTUTC']) # cconvert to datetime
        sen_df = sen_df.set_index('DTUTC') # Set index to datetime
        st.dataframe(sen_df, height = 200)
    else:
        st.info("Upload a CSV file to continue.")

    st.header("Upload Bottle File")
    # bottle_file = st.file_uploader("CSV file with headers: DTUTC, SPECPH...", type="csv")
    bottle_file = "/Users/taylorwirth/Desktop/ISFET_QC_GUI/bottle_example.csv"
    
    if bottle_file is not None:
        bott_df = pd.read_csv(bottle_file)
        bott_df['DTUTC'] = pd.to_datetime(bott_df['DTUTC']) # convert to datetime
        bott_df = bott_df.set_index('DTUTC') # Set index to datetime

        # interpolate bottle times and sensor vint
        bott_df['VINT'] = sen_df['VINT'].reindex(bott_df.index)
        bott_df = bott_df[bott_df['VINT'].notna()]

        bott_df['TCinsitu'] = sen_df['TEMPC'].reindex(bott_df.index)
        st.dataframe(bott_df, height = 200)
    else:
        st.info("Upload a CSV file to continue.")
        
    st.header("Upload Tris File (injection times)")
    # tris_file = st.file_uploader("CSV file with headers: ", type="csv")
    tris_file = "/Users/taylorwirth/Desktop/ISFET_QC_GUI/tris_example.csv"
    
    if tris_file is not None:
        tris_df = pd.read_csv(tris_file)
        tris_df['DTUTC'] = pd.to_datetime(tris_df['DTUTC']) # convert to datetime
        tris_df = tris_df.set_index('DTUTC') # Set index to datetime

        # interpolate tris times and sensor vint
        tris_df['VINT'] = sen_df['VINT'].reindex(tris_df.index)
        tris_df = tris_df[tris_df['VINT'].notna()] # remove rows where VINT is Nan

        # interpolate in situ temperature
        tris_df['TCinsitu'] = sen_df['TEMPC'].reindex(tris_df.index) 
        st.dataframe(tris_df, height = 200)
    else:
        st.info("Upload a CSV file to continue.")



with tab2:
    st.write("Validation samples were interpolated in time for Vint and in situ temperature.")

    # do this only if all files are uploaded
    if data_file is not None and tris_file is not None and bottle_file is not None:

        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

        # Vint subplot
        fig.add_trace(go.Scatter(x=sen_df.index, y=sen_df['VINT'],
                                    mode='lines', name='Vint', line=dict(color='black', width=1),
                                    showlegend=False),
                                    row=1, col=1)
        fig.add_trace(go.Scatter(x=tris_df.index, y=tris_df['VINT'], 
                                mode='markers', name='Tris', 
                                marker=dict(color="#00CBFE", symbol='triangle-up-open', size=10,
                                            line=dict(width=2))), 
                                row=1, col=1)
        fig.add_trace(go.Scatter(x=bott_df.index, y=bott_df['VINT'],
                                mode='markers', name='Bottle',
                                marker=dict(color="#FF33A7", symbol='circle-open', size=8,
                                            line=dict(width=2)),
                                            showlegend=True),
                                row=1, col=1)
        fig.update_yaxes(title_text="Vint (V)", row=1, col=1)

        # temperature subplot
        fig.add_trace(go.Scatter(x=sen_df.index, y=sen_df['TEMPC'],
                                 mode='lines', name=None, 
                                 line=dict(color='black', width=1),
                                 showlegend=False), 
                                 row=2, col=1)
        fig.add_trace(go.Scatter(x=tris_df.index, y=tris_df['TCinsitu'],
                                 mode='markers', name='Tris TempC',
                                 marker=dict(color='#00CBFE', symbol='triangle-up-open', size=10,
                                             line=dict(width=2)),
                                             showlegend=False),
                                row=2, col=1)
        fig.add_trace(go.Scatter(x=bott_df.index, y=bott_df['TCinsitu'],
                                 mode='markers', name='Bottle TempC',
                                 marker=dict(color="#FF33A7", symbol='circle-open', size=8,
                                             line=dict(width=2)),
                                             showlegend=False),
                                    row=2, col=1)
        fig.update_yaxes(title_text="Temperature (C)", row=2, col=1)
        
        fig.update_layout(height=600,dragmode='zoom')

        st.plotly_chart(fig, on_select="rerun", use_container_width=True)

    else:
        st.info("Upload a sensor data file to begin.")


with tab3:

    st.header("Calculate k0")

    # run_k0 = st.button("Run k0 Calculation")

    # if run_k0:
    
    # Set up a dict for the keyword arguments, for convenience
    pyco2_kws = {}

    # Define the known marine carbonate system parameters
    pyco2_kws["par1"] = bott_df['PHspec'] # pH measured in the lab, Total scale
    pyco2_kws["par2"] = bott_df['TA']  # TA measured in the lab in μmol/kg-sw
    pyco2_kws["par1_type"] = 3  # tell PyCO2SYS: "par1 is a pH value"
    pyco2_kws["par2_type"] = 1  # tell PyCO2SYS: "par2 is a TA value"

    # Define the seawater conditions and add them to the dict
    pyco2_kws["salinity"] = bott_df['SAL']  # practical salinity
    pyco2_kws["temperature"] = bott_df['TCspec']  # lab temperature (input conditions) in °C
    pyco2_kws["temperature_out"] = bott_df['TCinsitu']  # in-situ temperature (output conditions) in °C
    pyco2_kws["pressure"] = 0  # lab pressure (input conditions) in dbar, ignoring the atmosphere
    pyco2_kws["pressure_out"] = 0  # in-situ pressure (output conditions) in dbar, ignoring the atmosphere

    # Now calculate everything with PyCO2SYS!
    results = pyco2.sys(**pyco2_kws)
    bott_df['PHinsitu'] = results['pH_total_out']

    # calc in tris situ pH with DeValls & Dickson 1998
    S = 35
    T_K = tris_df['TCinsitu']+273.15

    # tris_df['PHinsitu'] = (11911.08 - 18.2499*S - 0.039336*S**2)/TK
    # + (-366.27059 + 0.53993607*S + 0.00016329*S**2)
    # + (64.52243 - 0.084041*S)*np.log(TK) - 0.11149858*TK
    tris_df['PHinsitu'] = (11911.08 - 18.2499*S - 0.039336*S**2)/T_K - \
    366.27059 + 0.53993607*S + 0.00016329*S**2 + \
    (64.52243 - 0.084041*S)*np.log(T_K) - 0.11149858*T_K
    
    # calculate kT and k0 for internal reference
    R = 8.31451; F = 96487; # Univ gas constant, Faraday constant, 
    TC = bott_df['TCinsitu']
    sal = bott_df['SAL']
    T_K = TC+273.15
    S_T = (R*T_K/F)*np.log(10)
    bott_df['kTint'] = bott_df['VINT']-bott_df['PHinsitu']*S_T

    k2int = -0.001455
    k0int_insitu = bott_df['VINT']-S_T*bott_df['PHinsitu']
    bott_df['k0int'] = k0int_insitu-k2int*(TC)

    TC = tris_df['TCinsitu']
    sal = 35
    T_K = TC+273.15
    S_T = (R*T_K/F)*np.log(10)
    tris_df['kTint'] = tris_df['VINT']-tris_df['PHinsitu']*S_T

    k2int = -0.001455
    k0int_insitu = tris_df['VINT']-S_T*tris_df['PHinsitu']
    tris_df['k0int'] = k0int_insitu-k2int*(TC)

    # # calculate kT for external reference
    # Z = 19.924*sal/(1000-1.005*sal) # Ionic strength, Dickson et al. 2007
    # SO4_tot = (0.14/96.062)*(sal/1.80655) # Total conservative sulfate
    # cCl = 0.99889/35.453*sal/1.80655 # Conservative chloride
    # mCl = cCl*1000/(1000-sal*35.165/35) # mol/kg-H2O
    # K_HSO4 = np.exp(-4276.1/T_K+141.328-23.093*np.log(T_K) + \
    #         (-13856/T_K+324.57-47.986*np.log(T_K))*Z**0.5 + \
    #         (35474/T_K-771.54+114.723*np.log(T_K))*Z-2698/T_K*Z**1.5 + \
    #         1776/T_K*Z**2+np.log(1-0.001005*sal)) # Bisulfate equilibrium const., Dickson et al. 2007
    # DHconst = 0.00000343*TC**2+0.00067524*TC+0.49172143 # Debye-Huckel, Khoo et al. 1977
    # log10gamma_HCl = 2*(-DHconst*np.sqrt(Z)/(1+1.394*np.sqrt(Z))+(0.08885-0.000111*TC)*Z)
    # SO4 = np.log10(1+SO4_tot/K_HSO4)
    # logsal = np.log10((1000-sal*35.165/35)/1000)

    # bott_df['kText'] = bott_df['VEXT'] - S*(bott_df['PHinsitu'] + \
    #                                   logsal + SO4 - np.log10(mCl) - log10gamma_HCl)

    

    # show validation samples and be able to edit QC
    # col1, col2 = st.columns(2) # Create two columns

    # with col1:
    #     st.subheader("Bottle samples")
    #     bott_edit = st.data_editor(
    #     bott_df,
    #     column_config={
    #         "QC": st.column_config.NumberColumn("QC", step=1)
    #     },
    #     use_container_width=False,
    #     num_rows="static"
    # )
    # with col2:
    #     st.subheader("Tris injections")
    #     tris_edit = st.data_editor(
    #     tris_df,
    #     column_config={
    #         "QC": st.column_config.NumberColumn("QC", step=1)
    #     },
    #     use_container_width=False,
    #     num_rows="static"
    # )
        
    # Reorder columns to show 'QC' first, next to the index
    def reorder_columns(df):
        cols = list(df.columns)
        if 'QC' in cols:
            cols.remove('QC')
            return ['QC'] + cols
        return cols

    # Prepare dataframes for editing
    bott_edit_df = bott_df[reorder_columns(bott_df)].copy()
    tris_edit_df = tris_df[reorder_columns(tris_df)].copy()

    # Show side-by-side editable QC columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<span style='color:#FF33A7; font-weight:bold; font-size:30px;'>Bottle samples</span>",
            unsafe_allow_html=True
        )
        edited_bott = st.data_editor(
            bott_edit_df,
            column_order=None,
            column_config={
                col: st.column_config.NumberColumn(disabled=(col != 'QC')) for col in bott_edit_df.columns
            },
            use_container_width=True,
            num_rows="dynamic",
            # height = 200
        )
    
    with col2:
        st.markdown(
            "<span style='color:#00CBFE; font-weight:bold; font-size:30px;'>Tris injections</span>",
            unsafe_allow_html=True
        )
        edited_tris = st.data_editor(
            tris_edit_df,
            column_order=None,
            column_config={
                col: st.column_config.NumberColumn(disabled=(col != 'QC')) for col in tris_edit_df.columns
            },
            use_container_width=True,
            num_rows="dynamic",
            # height = 200
        )

    # Save edited dataframes for later use
    bott_df_qc = edited_bott.copy()
    bott_df_qc = bott_df_qc[bott_df_qc['QC'] != 0]
    tris_df_qc = edited_tris.copy()
    tris_df_qc = tris_df_qc[tris_df_qc['QC'] != 0]
        
    # If the QC-ed dataframes exist and have changed, update the plots accordingly
    if 'bott_df_qc' in locals() and 'tris_df_qc' in locals():

        # Create subplots
        figkT = make_subplots(rows=1, cols=2, shared_xaxes=True, vertical_spacing=0.05)

        # kTint subplot
        figkT.add_trace( # k2 bottle scatter
            go.Scatter(
                x=bott_df_qc['TCinsitu'],
                y=bott_df_qc['kTint'],
                mode='markers',
                name='bottle',
                marker=dict(color="#FF33A7", symbol='circle-open', size=8, line=dict(width=2)),
                showlegend=True,
                customdata=bott_df_qc.index.strftime('%Y-%m-%d %H:%M:%S'),
                hovertemplate='TCinsitu: %{x}<br>kTint: %{y}<br>Date: %{customdata}<extra></extra>'),
        row=1, col=1)
        
        if len(bott_df_qc) > 1: # Linear fit for bottle samples (QC)
            x = bott_df_qc['TCinsitu']
            y = bott_df_qc['kTint']
            coeffs = np.polyfit(x, y, 1)
            fit_line = np.poly1d(coeffs)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = fit_line(x_fit)
            figkT.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    name='bottle fit',
                    line=dict(color="#FF33A7", dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
    
            slope = coeffs[0]
            intercept = coeffs[1]
            percent_diff = 100 * (slope - (-0.001455)) / abs(-0.001455)
            #text = f"Bottle Fit: y = {slope*1E6:.0f}uV {intercept:0.4f}"
            text = f"k2int = {slope*1E3:.3f} mV"
            text += f"<br>percent diff = {percent_diff:.2f}%"
            # Place annotation at the middle of the fit line
            x_annot = x_fit[len(x_fit)//2]
            y_annot = fit_line(x_annot)
            # Calculate angle of the fit line for annotation rotation
            # For visual angle correction based on plot scaling
            x_max = np.max(bott_df_qc['TCinsitu'])
            x_min = np.min(bott_df_qc['TCinsitu'])
            y_max = np.max(bott_df_qc['kTint'])
            y_min = np.min(bott_df_qc['kTint'])
            aspect_ratio = (x_max - x_min) / (y_max - y_min)

            # Rescale y difference
            y1 = bott_df_qc['kTint'].iloc[0]
            y2 = bott_df_qc['kTint'].iloc[-1]
            x1 = bott_df_qc['TCinsitu'].iloc[0]
            x2 = bott_df_qc['TCinsitu'].iloc[-1]
            dy = (y2 - y1) * aspect_ratio
            dx = (x2 - x1)

            visual_theta_rad = np.abs(math.atan2(dy, dx))
            visual_theta_deg = math.degrees(visual_theta_rad)
            angle_deg = visual_theta_deg
            figkT.add_annotation(
            x=x_annot,
            y=y_annot,
            text=text,
            showarrow=False,
            font=dict(color="#FF33A7"),
            yshift=20,
            xanchor="left",
            textangle=0
            )

        figkT.add_trace( # k2 tris scatter
            go.Scatter(
                x=tris_df_qc['TCinsitu'],
                y=tris_df_qc['kTint'],
                mode='markers',
                name='tris',
                marker=dict(color="#00CBFE", symbol='triangle-up-open', size=10, line=dict(width=2)),
                showlegend=True,
                customdata=tris_df_qc.index.strftime('%Y-%m-%d %H:%M:%S'),
                hovertemplate='TCinsitu: %{x}<br>kTint: %{y}<br>Date: %{customdata}<extra></extra>'),
            row=1, col=1)
        
        # Linear fit for tris  (QC)
        if len(tris_df_qc) > 1:
            x = tris_df_qc['TCinsitu']
            y = tris_df_qc['kTint']
            coeffs = np.polyfit(x, y, 1)
            fit_line = np.poly1d(coeffs)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = fit_line(x_fit)
            figkT.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    name='tris fit',
                    line=dict(color="#00CBFE", dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        if len(tris_df_qc) > 1: # Add annotation for tris linear fit
            slope = coeffs[0]
            intercept = coeffs[1]
            
            percent_diff = 100 * (slope - (-0.001455)) / abs(-0.001455)
            text = f"k2int = {slope*1E3:.3f} mV"
            text += f"<br>percent diff = {percent_diff:.2f}%"
            # Place annotation at the middle of the fit line
            x_annot = x_fit[len(x_fit)//2]
            y_annot = fit_line(x_annot)
            # Calculate angle of the fit line for annotation rotation
            # For visual angle correction based on plot scaling
            x_max = np.max(tris_df_qc['TCinsitu'])
            x_min = np.min(tris_df_qc['TCinsitu'])
            y_max = np.max(tris_df_qc['kTint'])
            y_min = np.min(tris_df_qc['kTint'])
            aspect_ratio = (x_max - x_min) / (y_max - y_min)

            # Rescale y difference
            y1 = tris_df_qc['kTint'].iloc[0]
            y2 = tris_df_qc['kTint'].iloc[-1]
            x1 = tris_df_qc['TCinsitu'].iloc[0]
            x2 = tris_df_qc['TCinsitu'].iloc[-1]
            dy = (y2 - y1) * aspect_ratio
            dx = (x2 - x1)

            visual_theta_rad = np.abs(math.atan2(dy, dx))
            visual_theta_deg = math.degrees(visual_theta_rad)
            angle_deg = visual_theta_deg
            figkT.add_annotation(
            x=x_annot,
            y=y_annot,
            text=text,
            showarrow=False,
            font=dict(color="#00CBFE"),
            yshift=20,
            xanchor="left",
            textangle=0
            )
        
        figkT.add_trace( # bottle k0int scatter
            go.Scatter(
                x=bott_df_qc.index,
                y=bott_df_qc['k0int'],
                mode='markers',
                name='bottle',
                marker=dict(color="#FF33A7", symbol='circle-open', size=8, line=dict(width=2)),
                showlegend=False,
                customdata=bott_df_qc.index.strftime('%Y-%m-%d %H:%M:%S'),
                hovertemplate='Date: %{x}<br>k0int: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        # Linear fit for k0int vs date (convert datetime to ordinal for fitting)
        if len(bott_df_qc) > 1:
            x_dates = bott_df_qc.index.map(pd.Timestamp.toordinal).values
            y_k0int = bott_df_qc['k0int'].values
            coeffs = np.polyfit(x_dates, y_k0int, 1)
            fit_line = np.poly1d(coeffs)
            x_fit = np.linspace(x_dates.min(), x_dates.max(), 100)
            y_fit = fit_line(x_fit)
            # Convert x_fit back to datetime for plotting
            x_fit_dates = [pd.Timestamp.fromordinal(int(x)) for x in x_fit]
            # figkT.add_trace(
            #     go.Scatter(
            #         x=x_fit_dates,
            #         y=y_fit,
            #         mode='lines',
            #         name='bottle fit',
            #         line=dict(color="#FF33A7", dash='dash'),
            #         showlegend=False
            #     ),
            #     row=1, col=2
            # )
            # Calculate slope per day
            figkT.add_hline(y=np.mean(bott_df_qc['k0int']), line_color='#FF33A7', line_dash='dash', line_width=1, row=1, col=2)

            slope_per_day = coeffs[0]
            text = f"k0int mean = {np.mean(bott_df_qc['k0int']):.4f} ± {np.std(bott_df_qc['k0int'])*1E6:.0f} uV"
            # text += f"<br>k0int drift = {slope_per_day*1E6:.0f} uV/day"
            # Place annotation at the middle of the fit line
            x_annot = x_fit_dates[len(x_fit_dates)//2]
            y_annot = fit_line(x_fit[len(x_fit)//2])
            y_annot = np.mean(bott_df_qc['k0int'])
            figkT.add_annotation(
                x=x_annot,
                y=y_annot,
                text=text,
                showarrow=False,
                font=dict(color="#FF33A7"),
                yshift=20,
                xanchor="left",
                yanchor="top",
                textangle=0,
                row=1, col=2
            )
        

        figkT.add_trace( # tris k0int scatter
            go.Scatter(
                x=tris_df_qc.index,
                y=tris_df_qc['k0int'],
                mode='markers',
                name='tris',
                marker=dict(color="#00CBFE", symbol='triangle-up-open', size=10, line=dict(width=2)),
                showlegend=False,
                customdata=tris_df_qc.index.strftime('%Y-%m-%d %H:%M:%S'),
                hovertemplate='Date: %{x}<br>k0int: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        # add horizontal line for mean tris k0int
        tris_mean = tris_df_qc['k0int'].mean()
        tris_std = tris_df_qc['k0int'].std()
        figkT.add_hline(y=tris_mean, line_color='black', line_dash='dash', line_width=1, row=1, col=2) 
        # add shaded box plus/minus 3 std for tris k0int
        figkT.add_shape(
           type="rect",
          x0=sen_df.index.min(), 
          x1=sen_df.index.max(),
          y0=tris_mean+0.0003,
          y1=tris_mean-0.0003,
         fillcolor="Gold",
        opacity=0.3,
        line_width=0,
          row=1, col=2
          )   

        # Linear fit for k0int vs date (convert datetime to ordinal for fitting)
        if len(tris_df_qc) > 1:
            x_dates = tris_df_qc.index.map(pd.Timestamp.toordinal).values
            y_k0int = tris_df_qc['k0int'].values
            coeffs = np.polyfit(x_dates, y_k0int, 1)
            fit_line = np.poly1d(coeffs)
            x_fit = np.linspace(x_dates.min(), x_dates.max(), 100)
            y_fit = fit_line(x_fit)
            # Convert x_fit back to datetime for plotting
            x_fit_dates = [pd.Timestamp.fromordinal(int(x)) for x in x_fit]
            figkT.add_trace(
                go.Scatter(
                    x=x_fit_dates,
                    y=y_fit,
                    mode='lines',
                    name='tris fit',
                    line=dict(color="#00CBFE", dash='dash'),
                    showlegend=False
                ),
                row=1, col=2
            )
            # Calculate slope per day
            slope_per_day = coeffs[0]
            text = f"k0int mean = {np.mean(tris_df_qc['k0int']):.4f} ± {np.std(tris_df_qc['k0int'])*1E6:.0f} uV"
            text += f"<br>k0int drift = {slope_per_day*1E6:.0f} uV/day"
            # Place annotation at the middle of the fit line
            x_annot = x_fit_dates[len(x_fit_dates)//2]
            y_annot = fit_line(x_fit[len(x_fit)//2])
            figkT.add_annotation(
                x=x_annot,
                y=y_annot,
                text=text,
                showarrow=False,
                font=dict(color="#00CBFE"),
                yshift=20,
                xanchor="left",
                textangle=0,
                row=1, col=2
            )
        


        figkT.update_yaxes(title_text="kTint (V)", row=1, col=1)
        figkT.update_xaxes(title_text="Temperature (C)", row=1, col=1)
        figkT.update_yaxes(title_text="k0int (V)", row=1, col=2)
        figkT.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
        figkT.update_layout(height=400,dragmode='zoom')
        st.plotly_chart(figkT, use_container_width=True)

        # plot k0 vs time

        # how to update 
        # else:
        #     st.warning("Press the RUN button")
        
with tab4: 
    st.header("Quality controlled pH data")
    # k0 option
    k0_option = st.radio(
        "Select k0 calculation method:",
        (
            "Bottle mean (QCed)",
            "Tris mean (QCed)",
            "Tris linear (QCed)"
        ),
        index=0
    )
    
    if k0_option == "Bottle mean (QCed)":
            k0int = bott_df_qc.loc[bott_df_qc['QC'] == 1, 'k0int'].mean()
    if k0_option == "Tris mean (QCed)":
            k0int = tris_df_qc.loc[tris_df_qc['QC'] == 1, 'k0int'].mean()
    if k0_option == "Tris linear (QCed)":
        # Calculate k0 from the linear fit of tris injections
        x_dates = tris_df_qc.index.map(pd.Timestamp.toordinal).values
        y_k0int = tris_df_qc['k0int'].values
        coeffs = np.polyfit(x_dates, y_k0int, 1)
        k0int = coeffs[1]
    st.write(f"Selected k0 value: {k0int:.6f} V")

    # calculate QCed pH from k0 option
    # Univ gas constant, Faraday constant, 
    R = 8.31451
    F = 96487
    # Temperature dependence of standard potentials, Martz et al. 2010
    k2int = -0.001455
    k2ext = -0.001048

    tempK = sen_df['TEMPC'] + 273.15  # Convert temp from C to 
    S_T = (R*tempK)/F*np.log(10) # Nernst temp dependence
    sen_df["pHint_cor"] = (sen_df['VINT']-(k0int+k2int*(sen_df["TEMPC"])))/S_T # Calc pHint from Nernst

    # Interpolate pHint_cor at the same times as the bottles and tris injections
    bott_df_qc['pHint_cor'] = sen_df['pHint_cor'].reindex(bott_df_qc.index)
    tris_df_qc['pHint_cor'] = sen_df['pHint_cor'].reindex(tris_df_qc.index)
 
    # calculate residuals (pHint_cor - pHinsitu)
    bott_df_qc['residuals'] = bott_df_qc['pHint_cor'] - bott_df_qc['PHinsitu']
    tris_df_qc['residuals'] = tris_df_qc['pHint_cor'] - tris_df_qc['PHinsitu']

    figpH = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                          subplot_titles=("Quality Controlled pH Data", "Temperature"))

    # pHint_cor subplot
    figpH.add_trace(
        go.Scatter(
            x=sen_df.index,
            y=sen_df['pHint_cor'],
            mode='lines',
            name='pHint corrected',
            marker=dict(color="#000000"),
            showlegend=False
        ),
        row=1, col=1
    )
    figpH.add_trace(
        go.Scatter(
            x=bott_df.index,
            y=bott_df['PHinsitu'],
            mode='markers',
            name='bottle',
            marker=dict(color="#FF33A7", symbol='circle-open', size=8, line=dict(width=2)),
            showlegend=True,
            customdata=bott_df.index.strftime('%Y-%m-%d %H:%M:%S'),
            hovertemplate='Date: %{customdata}<br>pH: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    figpH.add_trace(
        go.Scatter(
            x=tris_df.index,
            y=tris_df['PHinsitu'],
            mode='markers',
            name='tris',
            marker=dict(color="#00CBFE", symbol='triangle-up-open', size=10, line=dict(width=2)),
            showlegend=True,
            customdata=bott_df.index.strftime('%Y-%m-%d %H:%M:%S'),
            hovertemplate='Date: %{customdata}<br>pH: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    figpH.update_yaxes(title_text="pHint corrected", row=1, col=1)
    # adjsut x-axis range to the sensor data
    figpH.update_xaxes(range=[sen_df.index.min(), sen_df.index.max()], row=1, col=1)

    # Residual subplot
    # add horizontal black line at y=0 for the x-axis range of the residuals
     
 

    figpH.add_trace(
        go.Scatter(
            x=bott_df_qc.index,
            y=bott_df_qc['residuals'],
            mode='markers',
            name='bottle',
            marker=dict(color='#FF33A7', symbol='circle-open', size=8, line=dict(width=2)),
            showlegend=False
        ),
        row=2, col=1
    )
    figpH.add_hline(y=0, line_color='black', line_dash='dash', line_width=2, row=2, col=1)  
    # add shaded box plus/minus 0.006 pH for the residuals for the entire x-axis range
    figpH.add_shape(
        type="rect",
        x0=sen_df.index.min(),
        x1=sen_df.index.max(),
        y0=-0.006,
        y1=0.006,
        fillcolor="Gold",
        opacity=0.3,
        line_width=0,
        row=2, col=1
    )

    figpH.add_trace(
        go.Scatter(
            x=tris_df_qc.index,
            y=tris_df_qc['residuals'],
            mode='markers',
            name='tris',
            marker=dict(color='#00CBFE', symbol='triangle-up-open', size=10, line=dict(width=2)),
            showlegend=False,
            hovertemplate='Date: %{x}<br>resid: %{y:.4f}<extra></extra>'
            # hovertemplate y value formatted to 3 decimal places
            # hovertemplate='Date: %{x}<br>resid: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    figpH.update_xaxes(range=[sen_df.index.min(), sen_df.index.max()], row=2, col=1)
    figpH.update_yaxes(title_text="delta pH", row=2, col=1)

    figpH.update_xaxes(title_text="DateTime", row=2, col=1)
    figpH.update_layout(height=600)
    st.plotly_chart(figpH, use_container_width=True)



    


