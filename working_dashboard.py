#!/usr/bin/env python3
"""
Working Real-time Dashboard - Connects to running system
Author: Jay Guwalani
"""

import requests
import time
from datetime import datetime
import json

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.graph_objs as go
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    print("Install dash: pip install dash plotly")

class WorkingDashboard:
    """Dashboard that connects to live API"""
    
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        
        if not HAS_DASH:
            print("ERROR: dash not installed")
            return
        
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def get_metrics(self):
        """Fetch metrics from API"""
        try:
            response = requests.get(f"{self.api_url}/metrics", timeout=2)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching metrics: {e}")
        
        return {
            "token_metrics": {"total": 0, "cost_usd": 0, "by_agent": {}},
            "performance": {"requests": 0, "success_rate": 0, "avg_latency_ms": 0, "cache_hit_rate": 0},
            "system": {"available": False},
            "gpu": {"available": False}
        }
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Multi-Agent RAG Performance Dashboard", 
                   style={'textAlign': 'center'}),
            
            # Key Metrics Cards
            html.Div([
                html.Div([
                    html.H3("Total Requests"),
                    html.H2(id='total-requests', children='0')
                ], className='metric-card', style={'padding': '20px', 'margin': '10px', 'backgroundColor': '#f0f0f0'}),
                
                html.Div([
                    html.H3("Success Rate"),
                    html.H2(id='success-rate', children='0%')
                ], className='metric-card', style={'padding': '20px', 'margin': '10px', 'backgroundColor': '#f0f0f0'}),
                
                html.Div([
                    html.H3("Total Cost"),
                    html.H2(id='total-cost', children='$0.00')
                ], className='metric-card', style={'padding': '20px', 'margin': '10px', 'backgroundColor': '#f0f0f0'}),
                
                html.Div([
                    html.H3("Cache Hit Rate"),
                    html.H2(id='cache-rate', children='0%')
                ], className='metric-card', style={'padding': '20px', 'margin': '10px', 'backgroundColor': '#f0f0f0'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around'}),
            
            # Charts
            html.Div([
                html.Div([
                    dcc.Graph(id='token-chart')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='latency-chart')
                ], style={'width': '48%', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='agent-breakdown')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='cache-chart')
                ], style={'width': '48%', 'display': 'inline-block'}),
            ]),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=2000,  # 2 seconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('total-requests', 'children'),
             Output('success-rate', 'children'),
             Output('total-cost', 'children'),
             Output('cache-rate', 'children')],
            Input('interval-component', 'n_intervals')
        )
        def update_metrics(n):
            metrics = self.get_metrics()
            perf = metrics.get('performance', {})
            tokens = metrics.get('token_metrics', {})
            
            return (
                str(perf.get('requests', 0)),
                f"{perf.get('success_rate', 0):.1f}%",
                f"${tokens.get('cost_usd', 0):.4f}",
                f"{perf.get('cache_hit_rate', 0):.1f}%"
            )
        
        @self.app.callback(
            Output('token-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_token_chart(n):
            metrics = self.get_metrics()
            tokens = metrics.get('token_metrics', {})
            
            fig = go.Figure(data=[
                go.Bar(name='Prompt', y=['Tokens'], x=[tokens.get('prompt', 0)], orientation='h'),
                go.Bar(name='Completion', y=['Tokens'], x=[tokens.get('completion', 0)], orientation='h'),
                go.Bar(name='Cached', y=['Tokens'], x=[tokens.get('cached', 0)], orientation='h')
            ])
            
            fig.update_layout(
                title='Token Usage Breakdown',
                barmode='stack',
                height=300
            )
            
            return fig
        
        @self.app.callback(
            Output('latency-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_latency_chart(n):
            metrics = self.get_metrics()
            perf = metrics.get('performance', {})
            
            fig = go.Figure(data=[
                go.Indicator(
                    mode="gauge+number+delta",
                    value=perf.get('avg_latency_ms', 0),
                    title={'text': "Avg Latency (ms)"},
                    gauge={'axis': {'range': [None, 1000]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 200], 'color': "lightgreen"},
                               {'range': [200, 500], 'color': "yellow"},
                               {'range': [500, 1000], 'color': "red"}
                           ]},
                    domain={'x': [0, 1], 'y': [0, 1]}
                )
            ])
            
            fig.update_layout(height=300)
            return fig
        
        @self.app.callback(
            Output('agent-breakdown', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_agent_breakdown(n):
            metrics = self.get_metrics()
            by_agent = metrics.get('token_metrics', {}).get('by_agent', {})
            
            if not by_agent:
                by_agent = {'No data': 1}
            
            fig = go.Figure(data=[
                go.Pie(labels=list(by_agent.keys()), values=list(by_agent.values()))
            ])
            
            fig.update_layout(title='Token Usage by Agent', height=300)
            return fig
        
        @self.app.callback(
            Output('cache-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_cache_chart(n):
            metrics = self.get_metrics()
            perf = metrics.get('performance', {})
            cache_rate = perf.get('cache_hit_rate', 0)
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Hits', 'Misses'],
                    values=[cache_rate, 100-cache_rate],
                    marker_colors=['#2ecc71', '#e74c3c']
                )
            ])
            
            fig.update_layout(title='Cache Performance', height=300)
            return fig
    
    def run(self, debug=False, port=8050):
        """Run dashboard"""
        print(f"Starting dashboard on http://localhost:{port}")
        print(f"Connecting to API at {self.api_url}")
        self.app.run_server(debug=debug, port=port)


def main():
    """Main entry point"""
    import sys
    
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print("Multi-Agent RAG Dashboard")
    print("="*60)
    
    if not HAS_DASH:
        print("Install required packages:")
        print("  pip install dash plotly")
        return
    
    # Check if API is running
    try:
        response = requests.get(f"{api_url}/", timeout=2)
        if response.status_code == 200:
            print(f"✅ Connected to API at {api_url}")
        else:
            print(f"⚠️  API returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to API at {api_url}")
        print(f"   Error: {e}")
        print("\n   Start the API server first:")
        print("   python api_server.py")
        return
    
    dashboard = WorkingDashboard(api_url)
    dashboard.run()


if __name__ == "__main__":
    main()
