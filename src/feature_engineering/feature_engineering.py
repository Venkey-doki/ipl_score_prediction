"""
Feature Engineering Module for IPL Score Prediction

This module handles the creation and transformation of features for the IPL score prediction models.
It provides functions to generate team performance metrics, player statistics, venue characteristics,
and other relevant features for model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_team_performance_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that capture team performance over time.
    
    Args:
        matches_df: Cleaned matches DataFrame
        
    Returns:
        DataFrame with team performance features
    """
    logger.info("Creating team performance features")
    
    # Create a copy to avoid modifying the original
    df = matches_df.copy()
    
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Initialize features
    teams = pd.unique(df[['team1', 'team2']].values.ravel('K'))
    team_stats = {team: {'matches': 0, 'wins': 0, 'win_rate': 0.0} for team in teams}
    
    team_features = []
    
    # Calculate cumulative team performance
    for idx, row in df.iterrows():
        match_id = row['id']
        team1 = row['team1']
        team2 = row['team2']
        date = row['date']
        winner = row['winner']
        
        # Update team stats
        for team in [team1, team2]:
            team_stats[team]['matches'] += 1
            if team == winner:
                team_stats[team]['wins'] += 1
            
            # Calculate win rate
            matches = team_stats[team]['matches']
            wins = team_stats[team]['wins']
            win_rate = wins / matches if matches > 0 else 0
            team_stats[team]['win_rate'] = win_rate
        
        # Create features for this match
        team_features.append({
            'match_id': match_id,
            'date': date,
            'team1': team1,
            'team2': team2,
            'team1_matches': team_stats[team1]['matches'],
            'team1_wins': team_stats[team1]['wins'],
            'team1_win_rate': team_stats[team1]['win_rate'],
            'team2_matches': team_stats[team2]['matches'],
            'team2_wins': team_stats[team2]['wins'],
            'team2_win_rate': team_stats[team2]['win_rate']
        })
    
    team_performance_df = pd.DataFrame(team_features)
    logger.info(f"Created team performance features with {len(team_performance_df)} rows")
    
    return team_performance_df


def create_head_to_head_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that capture head-to-head performance between teams.
    
    Args:
        matches_df: Cleaned matches DataFrame
        
    Returns:
        DataFrame with head-to-head features
    """
    logger.info("Creating head-to-head features")
    
    # Create a copy to avoid modifying the original
    df = matches_df.copy()
    
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Initialize head-to-head records
    teams = pd.unique(df[['team1', 'team2']].values.ravel('K'))
    h2h_stats = {}
    
    for team1 in teams:
        for team2 in teams:
            if team1 != team2:
                h2h_stats[(team1, team2)] = {'matches': 0, 'wins': 0, 'win_rate': 0.0}
    
    h2h_features = []
    
    # Calculate cumulative head-to-head performance
    for idx, row in df.iterrows():
        match_id = row['id']
        team1 = row['team1']
        team2 = row['team2']
        date = row['date']
        winner = row['winner']
        
        # Update head-to-head stats
        h2h_stats[(team1, team2)]['matches'] += 1
        h2h_stats[(team2, team1)]['matches'] += 1
        
        if winner == team1:
            h2h_stats[(team1, team2)]['wins'] += 1
        elif winner == team2:
            h2h_stats[(team2, team1)]['wins'] += 1
        
        # Calculate win rates
        team1_h2h_matches = h2h_stats[(team1, team2)]['matches']
        team1_h2h_wins = h2h_stats[(team1, team2)]['wins']
        team1_h2h_win_rate = team1_h2h_wins / team1_h2h_matches if team1_h2h_matches > 0 else 0
        h2h_stats[(team1, team2)]['win_rate'] = team1_h2h_win_rate
        
        team2_h2h_matches = h2h_stats[(team2, team1)]['matches']
        team2_h2h_wins = h2h_stats[(team2, team1)]['wins']
        team2_h2h_win_rate = team2_h2h_wins / team2_h2h_matches if team2_h2h_matches > 0 else 0
        h2h_stats[(team2, team1)]['win_rate'] = team2_h2h_win_rate
        
        # Create features for this match
        h2h_features.append({
            'match_id': match_id,
            'date': date,
            'team1': team1,
            'team2': team2,
            'team1_h2h_matches': team1_h2h_matches,
            'team1_h2h_wins': team1_h2h_wins,
            'team1_h2h_win_rate': team1_h2h_win_rate,
            'team2_h2h_matches': team2_h2h_matches,
            'team2_h2h_wins': team2_h2h_wins,
            'team2_h2h_win_rate': team2_h2h_win_rate
        })
    
164|    h2h_df = pd.DataFrame(h2h_features)
165|    logger.info(f"Created head-to-head features with {len(h2h_df)} rows")
166|    
167|    return h2h_df
168|
169|
170|def create_venue_features(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
171|    """
172|    Create features that capture venue-specific characteristics and performance patterns.
173|    
174|    Args:
175|        matches_df: Cleaned matches DataFrame
176|        deliveries_df: Cleaned deliveries DataFrame
177|        
178|    Returns:
179|        DataFrame with venue-specific features
180|    """
181|    logger.info("Creating venue features")
182|    
183|    # Ensure the data is properly cleaned and prepared
184|    matches_df = matches_df.copy()
185|    
186|    # Get first innings scores by venue
187|    venue_first_innings = {}
188|    venue_chasing_results = {}
189|    venue_toss_decision = {}
190|    
191|    # Initialize dictionaries for venue statistics
192|    unique_venues = matches_df['venue'].unique()
193|    for venue in unique_venues:
194|        venue_first_innings[venue] = []
195|        venue_chasing_results[venue] = []
196|        venue_toss_decision[venue] = {'bat': 0, 'field': 0}
197|    
198|    # Process match data to extract venue-specific information
199|    for _, match in matches_df.iterrows():
200|        venue = match['venue']
201|        toss_decision = match.get('toss_decision', 'unknown')
202|        
203|        # Calculate first innings scores
204|        if 'target_runs' in match and not pd.isna(match['target_runs']) and match['target_runs'] > 0:
205|            venue_first_innings[venue].append(match['target_runs'])
206|        
207|        # Track toss decisions
208|        if toss_decision in ['bat', 'field']:
209|            venue_toss_decision[venue][toss_decision] += 1
210|        
211|        # Track chasing success
212|        if match['winner'] == match['team2'] and toss_decision == 'field':
213|            venue_chasing_results[venue].append(1)  # Successful chase
214|        elif match['winner'] == match['team1'] and toss_decision == 'bat':
215|            venue_chasing_results[venue].append(0)  # Unsuccessful chase
216|    
217|    # Create venue feature dataframe
218|    venue_features = []
219|    
220|    for venue in unique_venues:
221|        # Calculate first innings stats
222|        first_innings_scores = venue_first_innings[venue]
223|        avg_first_innings = np.mean(first_innings_scores) if first_innings_scores else 0
224|        max_first_innings = np.max(first_innings_scores) if first_innings_scores else 0
225|        min_first_innings = np.min(first_innings_scores) if first_innings_scores else 0
226|        
227|        # Calculate chasing success rate
228|        chasing_results = venue_chasing_results[venue]
229|        chase_success_rate = np.mean(chasing_results) if chasing_results else 0.5
230|        
231|        # Calculate toss preference
232|        bat_count = venue_toss_decision[venue]['bat']
233|        field_count = venue_toss_decision[venue]['field']
234|        toss_bat_preference = bat_count / (bat_count + field_count) if (bat_count + field_count) > 0 else 0.5
235|        
236|        # Create a feature row for this venue
237|        venue_features.append({
238|            'venue': venue,
239|            'avg_first_innings_score': avg_first_innings,
240|            'max_first_innings_score': max_first_innings,
241|            'min_first_innings_score': min_first_innings,
242|            'chase_success_rate': chase_success_rate,
243|            'toss_bat_preference': toss_bat_preference
244|        })
245|    
246|    venue_df = pd.DataFrame(venue_features)
247|    logger.info(f"Created venue features for {len(venue_df)} venues")
248|    
249|    return venue_df
250|
251|
252|def create_season_features(matches_df: pd.DataFrame) -> pd.DataFrame:
253|    """
254|    Create features that capture season-specific characteristics and trends.
255|    
256|    Args:
257|        matches_df: Cleaned matches DataFrame
258|        
259|    Returns:
260|        DataFrame with season-specific features
261|    """
262|    logger.info("Creating season features")
263|    
264|    # Ensure the data is properly cleaned and prepared
265|    matches_df = matches_df.copy()
266|    
267|    # Extract season from matches data
268|    if 'season' in matches_df.columns:
269|        season_col = 'season'
270|    elif 'date' in matches_df.columns:
271|        # Create a season column if it doesn't exist but we have date
272|        matches_df['date'] = pd.to_datetime(matches_df['date'])
273|        matches_df['season'] = matches_df['date'].dt.year
274|        season_col = 'season'
275|    else:
276|        logger.warning("No season or date column found in matches data")
277|        return pd.DataFrame()
278|    
279|    # Calculate season-level statistics
280|    season_stats = []
281|    
282|    for season, season_data in matches_df.groupby(season_col):
283|        total_matches = len(season_data)
284|        
285|        # Calculate average score in first innings
286|        first_innings_scores = []
287|        for _, match in season_data.iterrows():
288|            if 'target_runs' in match and not pd.isna(match['target_runs']) and match['target_runs'] > 0:
289|                first_innings_scores.append(match['target_runs'])
290|        
291|        avg_score = np.mean(first_innings_scores) if first_innings_scores else 0
292|        
293|        # Calculate win rates for chasing vs defending
294|        chasing_wins = 0
295|        defending_wins = 0
296|        toss_wins = 0
297|        
298|        for _, match in season_data.iterrows():
299|            if pd.isna(match['winner']) or match['winner'] == 'No Result':
300|                continue
301|                
302|            if 'toss_winner' in match.index and 'toss_decision' in match.index:
303|                toss_winner = match['toss_winner']
304|                toss_decision = match['toss_decision']
305|                winner = match['winner']
306|                
307|                # Check if toss winner won the match
308|                if toss_winner == winner:
309|                    toss_wins += 1
310|                
311|                # Check if chasing team won
312|                if (toss_winner == winner and toss_decision == 'field') or \
313|                   (toss_winner != winner and toss_decision == 'bat'):
314|                    chasing_wins += 1
315|                else:
316|                    defending_wins += 1
317|        
318|        # Calculate win rates
319|        chasing_win_rate = chasing_wins / (chasing_wins + defending_wins) if (chasing_wins + defending_wins) > 0 else 0.5
320|        toss_win_rate = toss_wins / total_matches if total_matches > 0 else 0.5
321|        
322|        # Add season stats
323|        season_stats.append({
324|            'season': season,
325|            'total_matches': total_matches,
326|            'avg_first_innings_score': avg_score,
327|            'chasing_win_rate': chasing_win_rate,
328|            'toss_win_rate': toss_win_rate
329|        })
330|    
331|    season_df = pd.DataFrame(season_stats)
332|    logger.info(f"Created season features for {len(season_df)} seasons")
333|    
334|    return season_df
335|
336|
337|def create_match_phase_features(deliveries_df: pd.DataFrame) -> pd.DataFrame:
338|    """
339|    Create features that capture performance during different phases of a match (e.g., powerplay, middle overs, death overs).
340|    
341|    Args:
342|        deliveries_df: Cleaned deliveries DataFrame
343|        
344|    Returns:
345|        DataFrame with match phase features
346|    """
347|    logger.info("Creating match phase features")
348|    
349|    # Create a copy to avoid modifying the original
350|    df = deliveries_df.copy()
351|    
352|    # Define match phases
353|    powerplay_overs = range(0, 6)
354|    middle_overs = range(6, 16)
355|    death_overs = range(16, 20)
356|    
357|    # Function to identify match phase
358|    def get_match_phase(over):
359|        if over in powerplay_overs:
360|            return 'powerplay'
361|        elif over in middle_overs:
362|            return 'middle'
363|        elif over in death_overs:
364|            return 'death'
365|        else:
366|            return 'other'
367|    
368|    # Add match phase column
369|    df['match_phase'] = df['over'].apply(get_match_phase)
370|    
371|    # Calculate phase-wise statistics for each match
372|    phase_stats = df.groupby(['match_id', 'inning', 'batting_team', 'match_phase']).agg({
373|        'total_runs': 'sum',
374|        'batsman_runs': 'sum',
375|        'is_wicket': 'sum',
376|        'ball': 'count'
377|    }).reset_index()
378|    
379|    # Calculate run rate for each phase
380|    phase_stats['balls'] = phase_stats['ball']
381|    phase_stats['overs'] = phase_stats['balls'] / 6
382|    phase_stats['run_rate'] = phase_stats['total_runs'] / phase_stats['overs']
383|    
384|    # Reshape to have one row per match-innings
385|    match_phases = []
386|    
387|    for (match_id, inning, batting_team), group in phase_stats.groupby(['match_id', 'inning', 'batting_team']):
388|        row = {
389|            'match_id': match_id,
390|            'inning': inning,
391|            'batting_team': batting_team
392|        }
393|        
394|        # Add stats for each phase
395|        for phase in ['powerplay', 'middle', 'death']:
396|            phase_data = group[group['match_phase'] == phase]
397|            
398|            if len(phase_data) > 0:
399|                row[f'{phase}_runs'] = phase_data['total_runs'].sum()
400|                row[f'{phase}_wickets'] = phase_data['is_wicket'].sum()
401|                row[f'{phase}_run_rate'] = phase_data['run_rate'].iloc[0]
402|            else:
403|                row[f'{phase}_runs'] = 0
404|                row[f'{phase}_wickets'] = 0
405|                row[f'{phase}_run_rate'] = 0
406|        
407|        match_phases.append(row)
408|    
409|    match_phase_df = pd.DataFrame(match_phases)
410|    logger.info(f"Created match phase features with {len(match_phase_df)} rows")
411|    
412|    return match_phase_df
413|
414|
415|def combine_all_features(
416|    matches_df: pd.DataFrame,
417|    deliveries_df: pd.DataFrame,
418|    for_prediction: bool = True
419|) -> pd.DataFrame:
420|    """
421|    Combine all feature sets into a single DataFrame for modeling.
422|    
423|    Args:
424|        matches_df: Cleaned matches DataFrame
425|        deliveries_df: Cleaned deliveries DataFrame
426|        for_prediction: If True, creates features for a predictive model,
427|                        otherwise creates features for exploratory analysis
428|        
429|    Returns:
430|        DataFrame with all combined features
431|    """
432|    logger.info("Combining all features")
433|    
434|    # Generate all feature sets
435|    team_perf = create_team_performance_features(matches_df)
436|    h2h = create_head_to_head_features(matches_df)
437|    venue_stats = create_venue_features(matches_df, deliveries_df)
438|    season_stats = create_season_features(matches_df)
439|    
440|    # Create base dataframe depending on the purpose
441|    if for_prediction:
442|        from ..data_processing.data_processing import prepare_score_prediction_data
443|        base_df = prepare_score_prediction_data(matches_df, deliveries_df)
444|    else:
445|        from ..data_processing.data_processing import join_match_and_deliveries
446|        base_df = join_match_and_deliveries(matches_df, deliveries_df)
447|        base_df = base_df.drop_duplicates(['match_id', 'inning']).reset_index(drop=True)
448|    
449|    # Merge team performance features
450|    if len(team_perf) > 0:
451|        base_df = pd.merge(
452|            base_df,
453|            team_perf,
454|            on='match_id',
455|            how='left'
456|        )
457|    
458|    # Merge venue features
459|    if len(venue_stats) > 0:
460|        base_df = pd.merge(
461|            base_df,
462|            venue_stats,
463|            on='venue',
464
