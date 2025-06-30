import json
from typing import List, Dict, Tuple, Set


class EntityEvaluator:
    """Evaluator for named entity and role extraction metrics."""

    def extract_entities_set(self, people_data: List[Dict]) -> Set[str]:
        """Extract entity names as a set for comparison."""
        return {person['name'].strip().lower() for person in people_data}
    
    def extract_entity_role_pairs(self, people_data: List[Dict]) -> Set[Tuple[str, str]]:
        """Extract (entity, role) pairs for role evaluation."""
        pairs = set()
        for person in people_data:
            name = person['name'].strip().lower()
            for role in person['roles']:
                pairs.add((name, role.strip().lower()))
        return pairs
    
    def calculate_precision_recall_f1(self, predicted: Set, ground_truth: Set) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1-score for sets."""
        if not predicted and not ground_truth:
            return 1.0, 1.0, 1.0
        
        if not predicted:
            return 0.0, 0.0, 0.0
        
        if not ground_truth:
            return 0.0, 0.0, 0.0
        
        intersection = predicted.intersection(ground_truth)
        
        precision = len(intersection) / len(predicted) if predicted else 0.0
        recall = len(intersection) / len(ground_truth) if ground_truth else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1
    
    def evaluate_entities(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Evaluate entity extraction performance.
        """

        # Mapping by article ID
        gt_dict = {item['uuid']: item for item in ground_truth}
        
        entity_precisions = []
        entity_recalls = []
        entity_f1s = []
        
        role_precisions = []
        role_recalls = []
        role_f1s = []
                
        for pred in predictions:
            if not pred['success'] or not pred['extraction']:
                continue
                
            article_id = pred['article_id']
            pred_data = pred['extraction']
            
            if article_id not in gt_dict:
                continue
                
            gt_data = gt_dict[article_id]
            
            # Entity evaluation: names
            pred_entities = self.extract_entities_set(pred_data['people'])
            gt_entities = self.extract_entities_set(gt_data['people'])
            
            ent_prec, ent_rec, ent_f1 = self.calculate_precision_recall_f1(pred_entities, gt_entities)
            entity_precisions.append(ent_prec)
            entity_recalls.append(ent_rec)
            entity_f1s.append(ent_f1)
            
            # Role evaluation: entity-role pairs
            pred_roles = self.extract_entity_role_pairs(pred_data['people'])
            gt_roles = self.extract_entity_role_pairs(gt_data['people'])
            
            role_prec, role_rec, role_f1 = self.calculate_precision_recall_f1(pred_roles, gt_roles)
            role_precisions.append(role_prec)
            role_recalls.append(role_rec)
            role_f1s.append(role_f1)
        
        # Aggregate results
        results = {
            'entity_precision': sum(entity_precisions) / len(entity_precisions) if entity_precisions else 0.0,
            'entity_recall': sum(entity_recalls) / len(entity_recalls) if entity_recalls else 0.0,
            'entity_f1': sum(entity_f1s) / len(entity_f1s) if entity_f1s else 0.0,
            'role_precision': sum(role_precisions) / len(role_precisions) if role_precisions else 0.0,
            'role_recall': sum(role_recalls) / len(role_recalls) if role_recalls else 0.0,
            'role_f1': sum(role_f1s) / len(role_f1s) if role_f1s else 0.0
        }
        
        return results
    
class TopicEvaluator:
    """Evaluator for topic and subtopic classification."""
    
    def evaluate_topics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Evaluate topic and subtopic classification performance.
        """
        gt_dict = {item['uuid']: item for item in ground_truth}
        
        topic_correct = 0
        subtopic_correct = 0
        total_matched = 0
        
        topic_details = []
        
        for pred in predictions:
            if not pred['success'] or not pred['extraction']:
                continue
                
            article_id = pred['article_id']
            pred_data = pred['extraction']
            
            if article_id not in gt_dict:
                continue
                
            gt_data = gt_dict[article_id]
            total_matched += 1
            
            # Topic accuracy
            pred_topic = pred_data['topic'].strip().lower()
            gt_topic = gt_data['topic'].strip().lower()
            topic_match = pred_topic == gt_topic
            
            if topic_match:
                topic_correct += 1
            
            # Subtopic accuracy
            pred_subtopic = pred_data['subtopic'].strip().lower()
            gt_subtopic = gt_data['subtopic'].strip().lower()
            subtopic_match = pred_subtopic == gt_subtopic
            
            if subtopic_match:
                subtopic_correct += 1
            
            topic_details.append({
                'article_id': article_id,
                'pred_topic': pred_data['topic'],
                'gt_topic': gt_data['topic'],
                'pred_subtopic': pred_data['subtopic'],
                'gt_subtopic': gt_data['subtopic'],
                'topic_match': topic_match,
                'subtopic_match': subtopic_match
            })
        
        results = {
            'topic_accuracy': topic_correct / total_matched if total_matched > 0 else 0.0,
            'subtopic_accuracy': subtopic_correct / total_matched if total_matched > 0 else 0.0,
            'total_matched': total_matched,
            'topic_details': topic_details
        }
        
        return results
    
def run_complete_evaluation(
    predictions_filepath: str,
    ground_truth_filepath: str,
    save_detailed_report: bool = True
) -> Dict:
    """
    Run complete evaluation and generate report.
    """
    print("Starting Final Evaluation")
    print("Loading data...")
    with open(predictions_filepath, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    with open(ground_truth_filepath, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    print(f"Loaded {len(ground_truth)} ground truth entries")
    
    print("\nEvaluating entity and role extraction...")
    entity_evaluator = EntityEvaluator()
    entity_results = entity_evaluator.evaluate_entities(predictions, ground_truth)
    
    print("Evaluating topic/subtopic classification...")
    topic_evaluator = TopicEvaluator()
    topic_results = topic_evaluator.evaluate_topics(predictions, ground_truth)
    
    # Combine results
    final_results = {
        'entity_metrics': entity_results,
        'topic_metrics': topic_results,
        'summary': {
            'entity_f1': entity_results['entity_f1'],
            'role_f1': entity_results['role_f1'],
            'topic_accuracy': topic_results['topic_accuracy'],
            'subtopic_accuracy': topic_results['subtopic_accuracy']
        }
    }
    
    print(f"\nEVALUATION RESULTS:")
    print(f"   Entity F1: {entity_results['entity_f1']:.3f}")
    print(f"   Role F1: {entity_results['role_f1']:.3f}")
    print(f"   Topic Accuracy: {topic_results['topic_accuracy']:.3f}")
    print(f"   Subtopic Accuracy: {topic_results['subtopic_accuracy']:.3f}")
    
    # Save report
    if save_detailed_report:
        report_path = "data/output/evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed report saved to: {report_path}")
    
    return final_results