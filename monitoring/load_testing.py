#!/usr/bin/env python3
"""
Load Testing Script for Enhanced Multi-Agent RAG System
Author: Jay Guwalani

Uses Locust for concurrent user simulation and performance benchmarking
Target: 2K-10K concurrent users with P50/P95/P99 latency tracking
"""

from locust import HttpUser, task, between, events
import random
import json
import time
from datetime import datetime
import statistics

# Global metrics storage
latency_metrics = {
    'p50': [],
    'p95': [],
    'p99': [],
    'successful': 0,
    'failed': 0,
    'total_latency': []
}


class MultiAgentRAGUser(HttpUser):
    """Simulated user for load testing"""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    
    # Sample queries for testing
    research_queries = [
        "What are the latest developments in LLM context windows?",
        "Explain the QLoRA fine-tuning approach",
        "Compare different multi-agent frameworks",
        "What is Retrieval Augmented Generation?",
        "How does semantic caching improve performance?"
    ]
    
    document_requests = [
        "Create an outline for a technical blog on AI",
        "Write a summary of machine learning trends",
        "Generate documentation for a Python project",
        "Create a technical report on RAG systems",
        "Draft a research paper outline"
    ]
    
    @task(3)  # 60% of requests
    def research_query(self):
        """Simulate research request"""
        query = random.choice(self.research_queries)
        
        start_time = time.time()
        
        with self.client.post(
            "/research",
            json={"query": query, "mode": "research_only"},
            catch_response=True
        ) as response:
            
            latency = time.time() - start_time
            latency_metrics['total_latency'].append(latency)
            
            if response.status_code == 200:
                latency_metrics['successful'] += 1
                response.success()
            else:
                latency_metrics['failed'] += 1
                response.failure(f"Status: {response.status_code}")
    
    @task(2)  # 40% of requests
    def document_request(self):
        """Simulate document creation request"""
        request = random.choice(self.document_requests)
        
        start_time = time.time()
        
        with self.client.post(
            "/document",
            json={"request": request, "mode": "document_only"},
            catch_response=True
        ) as response:
            
            latency = time.time() - start_time
            latency_metrics['total_latency'].append(latency)
            
            if response.status_code == 200:
                latency_metrics['successful'] += 1
                response.success()
            else:
                latency_metrics['failed'] += 1
                response.failure(f"Status: {response.status_code}")
    
    @task(1)  # 20% of requests
    def full_workflow(self):
        """Simulate full workflow request"""
        query = "Research AI trends and create a comprehensive report"
        
        start_time = time.time()
        
        with self.client.post(
            "/process",
            json={"message": query, "mode": "full"},
            catch_response=True
        ) as response:
            
            latency = time.time() - start_time
            latency_metrics['total_latency'].append(latency)
            
            if response.status_code == 200:
                latency_metrics['successful'] += 1
                response.success()
            else:
                latency_metrics['failed'] += 1
                response.failure(f"Status: {response.status_code}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Calculate and display final metrics"""
    
    if latency_metrics['total_latency']:
        sorted_latencies = sorted(latency_metrics['total_latency'])
        n = len(sorted_latencies)
        
        p50 = sorted_latencies[int(n * 0.50)]
        p95 = sorted_latencies[int(n * 0.95)]
        p99 = sorted_latencies[int(n * 0.99)]
        
        print("\n" + "="*60)
        print("LOAD TEST RESULTS - Multi-Agent RAG System")
        print("="*60)
        print(f"Total Requests: {latency_metrics['successful'] + latency_metrics['failed']}")
        print(f"Successful: {latency_metrics['successful']}")
        print(f"Failed: {latency_metrics['failed']}")
        print(f"Success Rate: {(latency_metrics['successful'] / max(latency_metrics['successful'] + latency_metrics['failed'], 1)) * 100:.2f}%")
        print(f"\nLatency Metrics:")
        print(f"  P50 (median): {p50:.3f}s")
        print(f"  P95: {p95:.3f}s")
        print(f"  P99: {p99:.3f}s")
        print(f"  Average: {statistics.mean(sorted_latencies):.3f}s")
        print(f"  Min: {min(sorted_latencies):.3f}s")
        print(f"  Max: {max(sorted_latencies):.3f}s")
        print("="*60)


# Benchmark test scenarios
class BenchmarkScenarios:
    """Different load testing scenarios"""
    
    @staticmethod
    def light_load():
        """Light load: 100 concurrent users"""
        return {
            'users': 100,
            'spawn_rate': 10,
            'run_time': '5m'
        }
    
    @staticmethod
    def moderate_load():
        """Moderate load: 1000 concurrent users"""
        return {
            'users': 1000,
            'spawn_rate': 50,
            'run_time': '10m'
        }
    
    @staticmethod
    def heavy_load():
        """Heavy load: 5000 concurrent users"""
        return {
            'users': 5000,
            'spawn_rate': 100,
            'run_time': '15m'
        }
    
    @staticmethod
    def stress_test():
        """Stress test: 10000 concurrent users"""
        return {
            'users': 10000,
            'spawn_rate': 200,
            'run_time': '20m'
        }


def run_benchmark(scenario='moderate'):
    """Run a specific benchmark scenario"""
    
    scenarios = {
        'light': BenchmarkScenarios.light_load(),
        'moderate': BenchmarkScenarios.moderate_load(),
        'heavy': BenchmarkScenarios.heavy_load(),
        'stress': BenchmarkScenarios.stress_test()
    }
    
    config = scenarios.get(scenario, BenchmarkScenarios.moderate_load())
    
    print(f"\nRunning {scenario.upper()} load test:")
    print(f"  Users: {config['users']}")
    print(f"  Spawn Rate: {config['spawn_rate']}/s")
    print(f"  Duration: {config['run_time']}")
    print("\nStarting test...\n")
    
    # In practice, this would be run via command line:
    # locust -f load_testing.py --users 1000 --spawn-rate 50 --run-time 10m
    

def main():
    """Main entry point"""
    import sys
    
    print("Multi-Agent RAG System - Load Testing Tool")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        run_benchmark(scenario)
    else:
        print("\nUsage:")
        print("  python load_testing.py [scenario]")
        print("\nScenarios:")
        print("  light    - 100 concurrent users")
        print("  moderate - 1000 concurrent users")
        print("  heavy    - 5000 concurrent users")
        print("  stress   - 10000 concurrent users")
        print("\nOr run with Locust directly:")
        print("  locust -f load_testing.py --users 1000 --spawn-rate 50 --run-time 10m --host http://localhost:8000")


if __name__ == "__main__":
    main()
