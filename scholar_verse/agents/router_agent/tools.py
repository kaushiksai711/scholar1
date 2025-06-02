from typing import Dict, Any, List, Optional
from datetime import datetime
from google.adk.tools.tool_context import ToolContext

def analyze_document_complexity(tool_context: ToolContext, document_content: str) -> Dict[str, Any]:
    """
    Analyzes the complexity of a document and updates the state with analysis results.
    
    Args:
        tool_context: The context containing state and other utilities
        document_content: The content of the document to analyze
        
    Returns:
        Dict containing analysis results and status
    """
    # Initialize state if needed
    if 'document_status' not in tool_context.state:
        tool_context.state['document_status'] = {
            'ingested': False,
            'processed': False,
            'analyzed': False
        }
    if 'analysis_results' not in tool_context.state:
        tool_context.state['analysis_results'] = {}
    if 'interaction_history' not in tool_context.state:
        tool_context.state['interaction_history'] = []

    # Simple complexity analysis (can be enhanced with more sophisticated analysis)
    word_count = len(document_content.split())
    paragraph_count = len([p for p in document_content.split('\n\n') if p.strip()])
    has_references = 'references' in document_content.lower() or 'bibliography' in document_content.lower()
    
    # Calculate complexity score (simplified)
    complexity_score = 0
    if word_count > 10000:
        complexity_score += 2
    elif word_count > 5000:
        complexity_score += 1
        
    if has_references:
        complexity_score += 1
        
    if paragraph_count > 50:
        complexity_score += 1
    
    complexity_level = 'low'
    if complexity_score >= 3:
        complexity_level = 'high'
    elif complexity_score >= 1:
        complexity_level = 'medium'

    # Update state with analysis results
    tool_context.state['analysis_results'].update({
        'word_count': word_count,
        'paragraph_count': paragraph_count,
        'has_references': has_references,
        'complexity_score': complexity_score,
        'complexity_level': complexity_level,
        'last_analyzed': datetime.now().isoformat()
    })
    
    tool_context.state['document_status']['analyzed'] = True
    
    # Log this interaction
    tool_context.state['interaction_history'].append({
        'action': 'document_analysis',
        'timestamp': datetime.now().isoformat(),
        'details': {
            'complexity_level': complexity_level,
            'metrics': {
                'word_count': word_count,
                'paragraph_count': paragraph_count,
                'has_references': has_references
            }
        }
    })
    
    return {
        'status': 'success',
        'complexity_level': complexity_level,
        'metrics': {
            'word_count': word_count,
            'paragraph_count': paragraph_count,
            'has_references': has_references
        }
    }

def route_to_appropriate_agent(tool_context: ToolContext, query: str, document_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Determines the most appropriate sub-agent to handle the given query or document.
    
    Args:
        tool_context: The context containing state and other utilities
        query: The user's query to be routed
        document_path: Optional path to a document that needs processing
        
    Returns:
        Dict containing routing decision and target agent
    """
    # Initialize state if needed
    if 'routing_history' not in tool_context.state:
        tool_context.state['routing_history'] = []
    
    # Initialize state if needed
    if 'routing_history' not in tool_context.state:
        tool_context.state['routing_history'] = []
    
    # Check for document upload/ingestion request
    if document_path or any(term in query.lower() for term in ['ingest', 'upload', 'process', 'document', 'paper', 'pdf']):
        return {
            'status': 'success',
            'routing_decision': {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'document_path': document_path,
                'target_agent': 'ingestion_agent',
                'confidence': 'high',
                'reasoning': 'Document processing request detected',
                'action_required': 'process_document',
                'parameters': {
                    'document_path': document_path,
                    'query': query if query != 'ingest' else None
                }
            }
        }
    
    # Simple keyword-based routing (can be enhanced with ML/NLP)
    query_lower = query.lower()
    routing_decision = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'confidence': 'medium',
        'target_agent': None,
        'reasoning': '',
        'action_required': None,
        'parameters': {}
    }
    
    # Routing logic for non-document queries
    if any(term in query_lower for term in ['cite', 'reference', 'citation', 'graph']):
        routing_decision.update({
            'target_agent': 'citation_graph_agent',
            'confidence': 'high',
            'reasoning': 'Query relates to citations or reference networks',
            'action_required': 'analyze_citations'
        })
    elif any(term in query for term in ['cite', 'reference', 'citation', 'graph']):
        routing_decision.update({
            'target_agent': 'citation_graph_agent',
            'confidence': 'high',
            'reasoning': 'Query relates to citations or reference networks'
        })
    elif any(term in query for term in ['compare', 'across papers', 'trend', 'methodology']):
        routing_decision.update({
            'target_agent': 'cross_paper_analysis_agent',
            'confidence': 'high',
            'reasoning': 'Query requires comparative analysis across papers'
        })
    elif any(term in query for term in ['search', 'find', 'look up', 'web']):
        routing_decision.update({
            'target_agent': 'deep_search_agent',
            'confidence': 'high',
            'reasoning': 'Query requires external information retrieval'
        })
    elif any(term in query for term in ['summarize', 'insight', 'analyze', 'findings']):
        routing_decision.update({
            'target_agent': 'insight_agent',
            'confidence': 'high',
            'reasoning': 'Query requires generating insights or summaries'
        })
    elif any(term in query for term in ['visualize', 'graph', 'diagram', 'chart']):
        routing_decision.update({
            'target_agent': 'visualization_agent',
            'confidence': 'high',
            'reasoning': 'Query requires data visualization'
        })
    else:
        # Default to ingestion agent if no clear match
        routing_decision.update({
            'target_agent': 'ingestion_agent',
            'confidence': 'low',
            'reasoning': 'No specific agent matched, defaulting to ingestion agent'
        })
    
    # Log the routing decision
    tool_context.state['routing_history'].append(routing_decision)
    
    # Update interaction history
    tool_context.state['interaction_history'].append({
        'action': 'routing_decision',
        'timestamp': datetime.now().isoformat(),
        'details': routing_decision
    })
    
    return {
        'status': 'success',
        'routing_decision': routing_decision
    }

def evaluate_agent_performance(tool_context: ToolContext, 
                             agent_name: str, 
                             metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tracks and evaluates the performance of sub-agents.
    
    Args:
        tool_context: The context containing state and other utilities
        agent_name: Name of the agent being evaluated
        metrics: Dictionary of performance metrics
        
    Returns:
        Dict containing evaluation results
    """
    # Initialize metrics tracking if needed
    if 'agent_metrics' not in tool_context.state:
        tool_context.state['agent_metrics'] = {}
    if agent_name not in tool_context.state['agent_metrics']:
        tool_context.state['agent_metrics'][agent_name] = {
            'invocation_count': 0,
            'success_count': 0,
            'error_count': 0,
            'average_processing_time': 0,
            'last_invoked': None,
            'metrics_history': []
        }
    
    # Get agent metrics
    agent_metrics = tool_context.state['agent_metrics'][agent_name]
    
    # Update metrics
    agent_metrics['invocation_count'] += 1
    agent_metrics['last_invoked'] = datetime.now().isoformat()
    
    if metrics.get('status') == 'success':
        agent_metrics['success_count'] += 1
    else:
        agent_metrics['error_count'] += 1
    
    # Update average processing time (weighted average)
    if 'processing_time' in metrics:
        current_avg = agent_metrics['average_processing_time']
        count = agent_metrics['success_count'] + agent_metrics['error_count']
        agent_metrics['average_processing_time'] = (
            (current_avg * (count - 1) + metrics['processing_time']) / count
        )
    
    # Store historical metrics
    metrics_entry = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'agent_state': {
            'invocation_count': agent_metrics['invocation_count'],
            'success_count': agent_metrics['success_count'],
            'error_count': agent_metrics['error_count']
        }
    }
    agent_metrics['metrics_history'].append(metrics_entry)
    
    # Log this evaluation
    tool_context.state['interaction_history'].append({
        'action': 'agent_performance_evaluation',
        'timestamp': datetime.now().isoformat(),
        'agent': agent_name,
        'metrics': metrics
    })
    
    return {
        'status': 'success',
        'agent': agent_name,
        'metrics': agent_metrics
    }

def process_user_feedback(tool_context: ToolContext, 
                         feedback: str, 
                         rating: int,
                         context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes user feedback and updates agent behavior accordingly.
    
    Args:
        tool_context: The context containing state and other utilities
        feedback: User's feedback text
        rating: Numeric rating (e.g., 1-5)
        context: Additional context about the feedback
        
    Returns:
        Dict containing processing status and any updates made
    """
    # Initialize feedback tracking if needed
    if 'user_feedback' not in tool_context.state:
        tool_context.state['user_feedback'] = []
    
    # Create feedback entry
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'feedback': feedback,
        'rating': rating,
        'context': context
    }
    
    # Store feedback
    tool_context.state['user_feedback'].append(feedback_entry)
    
    # Update user preferences based on feedback (if applicable)
    if 'user_preferences' not in tool_context.state:
        tool_context.state['user_preferences'] = {}
    
    # Example: If feedback mentions analysis depth, update preferences
    if 'analysis depth' in feedback.lower():
        if 'deeper' in feedback.lower() or 'more detailed' in feedback.lower():
            tool_context.state['user_preferences']['analysis_depth'] = 'advanced'
        elif 'simpler' in feedback.lower() or 'less detailed' in feedback.lower():
            tool_context.state['user_preferences']['analysis_depth'] = 'basic'
    
    # Log this interaction
    tool_context.state['interaction_history'].append({
        'action': 'user_feedback_received',
        'timestamp': datetime.now().isoformat(),
        'details': {
            'rating': rating,
            'feedback_type': 'explicit',
            'preferences_updated': 'analysis_depth' in feedback.lower()
        }
    })
    
    return {
        'status': 'success',
        'feedback_processed': True,
        'preferences_updated': tool_context.state.get('user_preferences', {})
    }

