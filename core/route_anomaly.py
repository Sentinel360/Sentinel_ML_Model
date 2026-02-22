"""
Route Anomaly Detection System
Detects when driver deviates from expected routes
"""

import googlemaps
import numpy as np
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform
import pyproj

class RouteAnomalyDetector:
    """
    Real-time route deviation detection
    
    Handles:
    - Multiple valid routes
    - Dynamic rerouting
    - Traffic-aware updates
    - Direction-based anomaly detection
    """
    
    def __init__(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        google_api_key: str,
        buffer_distance: int = 100  # meters
    ):
        self.origin = origin
        self.destination = destination
        self.api_key = google_api_key
        self.buffer_distance = buffer_distance
        
        # Initialize Google Maps client
        self.gmaps = googlemaps.Client(key=self.api_key)
        
        # Fetch all valid routes
        self.routes = self._fetch_all_routes()
        self.primary_route = self.routes[0] if self.routes else None
        self.alternative_routes = self.routes[1:] if len(self.routes) > 1 else []
        
        # Create route corridors (buffers)
        self.route_corridors = self._create_route_corridors()
        
        # Tracking
        self.trip_start_time = datetime.now().timestamp()
        self.last_reroute_check = self.trip_start_time
        self.gps_breadcrumbs = []
        self.deviation_events = []
        self.consecutive_deviations = 0
        
    def _fetch_all_routes(self) -> List[Dict]:
        """
        Get all valid routes from Google Maps
        
        Returns list of routes, each containing:
        - polyline: List of (lat, lon) points
        - distance: meters
        - duration: seconds
        - summary: Text description (e.g., "via Madina")
        """
        try:
            directions = self.gmaps.directions(
                origin=self.origin,
                destination=self.destination,
                mode='driving',
                departure_time='now',  # Traffic-aware
                alternatives=True,     # Get multiple routes
                traffic_model='best_guess'
            )
            
            if not directions:
                raise Exception("No routes found from Google Maps")
            
            routes = []
            for route_data in directions:
                # Decode polyline
                polyline_encoded = route_data['overview_polyline']['points']
                polyline_points = googlemaps.convert.decode_polyline(polyline_encoded)
                
                routes.append({
                    'polyline': polyline_points,
                    'distance': route_data['legs'][0]['distance']['value'],
                    'duration': route_data['legs'][0]['duration']['value'],
                    'summary': route_data['summary'],
                    'bounds': route_data['bounds']
                })
            
            return routes
            
        except Exception as e:
            print(f"Error fetching routes: {e}")
            return []
    
    def _create_route_corridors(self) -> List[Polygon]:
        """
        Create buffer polygons around each route
        """
        corridors = []
        
        for route in self.routes:
            # Create LineString from polyline
            line = LineString([(p[1], p[0]) for p in route['polyline']])  # lon, lat
            
            # Transform to UTM for accurate meter-based buffer
            # Ghana is in UTM Zone 30N
            wgs84 = pyproj.CRS('EPSG:4326')
            utm = pyproj.CRS('EPSG:32630')  # UTM Zone 30N
            
            project_to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)
            project_to_wgs84 = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True)
            
            # Transform to UTM, buffer, transform back
            line_utm = transform(project_to_utm.transform, line)
            buffer_utm = line_utm.buffer(self.buffer_distance)
            buffer_wgs84 = transform(project_to_wgs84.transform, buffer_utm)
            
            corridors.append(buffer_wgs84)
        
        return corridors
    
    def check_for_reroutes(self, current_position: Tuple[float, float]):
        """
        Check if Google Maps has rerouted due to traffic
        Call every 30 seconds
        """
        current_time = datetime.now().timestamp()
        
        # Only check every 30 seconds
        if current_time - self.last_reroute_check < 30:
            return
        
        # Fetch fresh routes from current position
        try:
            updated_routes_data = self.gmaps.directions(
                origin=current_position,
                destination=self.destination,
                mode='driving',
                departure_time='now',
                alternatives=True,
                traffic_model='best_guess'
            )
            
            if updated_routes_data:
                # Parse updated routes
                updated_routes = []
                for route_data in updated_routes_data:
                    polyline_points = googlemaps.convert.decode_polyline(
                        route_data['overview_polyline']['points']
                    )
                    updated_routes.append({
                        'polyline': polyline_points,
                        'distance': route_data['legs'][0]['distance']['value'],
                        'duration': route_data['legs'][0]['duration']['value'],
                        'summary': route_data['summary']
                    })
                
                # Update routes and corridors
                self.routes = updated_routes
                self.route_corridors = self._create_route_corridors()
                
                self.last_reroute_check = current_time
                
                return {
                    'rerouted': True,
                    'new_route': updated_routes[0]['summary']
                }
        
        except Exception as e:
            print(f"Reroute check failed: {e}")
        
        self.last_reroute_check = current_time
        return {'rerouted': False}
    
    def haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two GPS points (meters)"""
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi/2)**2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda/2)**2
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def distance_from_route(
        self,
        current_pos: Tuple[float, float],
        route_polyline: List[Tuple[float, float]]
    ) -> Dict:
        """
        Find minimum distance from current position to route
        """
        min_distance = float('inf')
        closest_point = None
        segment_index = 0
        
        current_lat, current_lon = current_pos
        
        for i in range(len(route_polyline) - 1):
            p1 = route_polyline[i]
            p2 = route_polyline[i + 1]
            
            # Find closest point on this segment
            closest_on_segment = self._closest_point_on_segment(
                current_pos, p1, p2
            )
            
            # Calculate distance
            dist = self.haversine_distance(
                current_lat, current_lon,
                closest_on_segment[0], closest_on_segment[1]
            )
            
            if dist < min_distance:
                min_distance = dist
                closest_point = closest_on_segment
                segment_index = i
        
        return {
            'distance': min_distance,
            'closest_point': closest_point,
            'segment_index': segment_index,
            'progress': segment_index / max(len(route_polyline) - 1, 1)
        }
    
    def _closest_point_on_segment(
        self,
        point: Tuple[float, float],
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Find closest point on line segment to given point"""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return seg_start
        
        fx = px - x1
        fy = py - y1
        
        t = max(0, min(1, (fx*dx + fy*dy) / (dx*dx + dy*dy)))
        
        return (x1 + t*dx, y1 + t*dy)
    
    def calculate_bearing(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """
        Calculate bearing from point1 to point2 (degrees)
        0° = North, 90° = East, 180° = South, 270° = West
        """
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlon = lon2 - lon1
        
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - \
            math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def is_within_any_corridor(
        self,
        current_pos: Tuple[float, float]
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if position is within any route corridor
        
        Returns: (is_within, route_index)
        """
        point = Point(current_pos[1], current_pos[0])  # lon, lat
        
        for idx, corridor in enumerate(self.route_corridors):
            if corridor.contains(point):
                return (True, idx)
        
        return (False, None)
    
    def update(
        self,
        current_gps: Tuple[float, float],
        timestamp: float
    ) -> Dict:
        """
        Process new GPS point
        
        Returns detection result with risk adjustment
        """
        
        # Store breadcrumb
        self.gps_breadcrumbs.append({
            'position': current_gps,
            'timestamp': timestamp
        })
        
        # Check for reroutes every 30 seconds
        self.check_for_reroutes(current_gps)
        
        # Check if within any corridor
        within_corridor, route_idx = self.is_within_any_corridor(current_gps)
        
        if within_corridor:
            # On valid route
            self.consecutive_deviations = 0
            
            route_name = self.routes[route_idx]['summary']
            is_primary = (route_idx == 0)
            
            return {
                'status': 'ON_PRIMARY_ROUTE' if is_primary else 'ON_ALTERNATIVE_ROUTE',
                'route': route_name,
                'deviation_distance': 0.0,
                'risk_adjustment': 0.0,
                'triggered': False
            }
        
        # Not in any corridor - calculate minimum deviation
        min_deviation_info = None
        min_distance = float('inf')
        
        for route in self.routes:
            dev_info = self.distance_from_route(current_gps, route['polyline'])
            if dev_info['distance'] < min_distance:
                min_distance = dev_info['distance']
                min_deviation_info = dev_info
        
        # Track consecutive deviations
        self.consecutive_deviations += 1
        
        # Record deviation event
        self.deviation_events.append({
            'timestamp': timestamp,
            'distance': min_distance,
            'position': current_gps
        })
        
        # Direction check (if we have previous position)
        wrong_direction = False
        if len(self.gps_breadcrumbs) >= 2:
            bearing_to_dest = self.calculate_bearing(current_gps, self.destination)
            current_heading = self.calculate_bearing(
                self.gps_breadcrumbs[-2]['position'],
                current_gps
            )
            
            angle_diff = abs(bearing_to_dest - current_heading)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff > 90:  # Heading perpendicular or opposite
                wrong_direction = True
        
        # Evaluate severity
        return self._evaluate_deviation(
            min_distance,
            self.consecutive_deviations,
            wrong_direction,
            timestamp
        )
    
    def _evaluate_deviation(
        self,
        distance: float,
        consecutive: int,
        wrong_direction: bool,
        timestamp: float
    ) -> Dict:
        """
        Evaluate deviation severity and return risk adjustment
        """
        
        # CRITICAL: Wrong direction
        if wrong_direction and distance > 300:
            return {
                'status': 'WRONG_DIRECTION',
                'deviation_distance': distance,
                'consecutive_seconds': consecutive,
                'risk_adjustment': 0.60,
                'triggered': True,
                'severity': 'CRITICAL',
                'action': 'ALERT_USER_IMMEDIATELY',
                'message': '⚠️ Driver heading away from destination!'
            }
        
        # CRITICAL: Large sustained deviation
        if distance > 500 and consecutive >= 5:
            return {
                'status': 'CRITICAL_DEVIATION',
                'deviation_distance': distance,
                'consecutive_seconds': consecutive,
                'risk_adjustment': 0.40,
                'triggered': True,
                'severity': 'CRITICAL',
                'action': 'ALERT_USER_IMMEDIATELY',
                'message': f'⚠️ {distance:.0f}m off route for {consecutive}s'
            }
        
        # HIGH: Moderate sustained deviation
        if distance > 200 and consecutive >= 10:
            return {
                'status': 'HIGH_DEVIATION',
                'deviation_distance': distance,
                'consecutive_seconds': consecutive,
                'risk_adjustment': 0.25,
                'triggered': True,
                'severity': 'HIGH',
                'action': 'CHECK_IN_WITH_USER',
                'message': 'Driver taking different route. Are you okay?'
            }
        
        # MEDIUM: Prolonged minor deviation
        if distance > 150 and consecutive >= 20:
            return {
                'status': 'PROLONGED_DEVIATION',
                'deviation_distance': distance,
                'consecutive_seconds': consecutive,
                'risk_adjustment': 0.15,
                'triggered': True,
                'severity': 'MEDIUM',
                'action': 'MONITOR',
                'message': 'Route deviation detected'
            }
        
        # LOW: Brief deviation (likely avoiding obstacle, parking)
        return {
            'status': 'MINOR_DEVIATION',
            'deviation_distance': distance,
            'consecutive_seconds': consecutive,
            'risk_adjustment': 0.05,
            'triggered': False,
            'severity': 'LOW',
            'action': None
        }
    
    def get_trip_summary(self) -> Dict:
        """Post-trip analysis"""
        if len(self.gps_breadcrumbs) == 0:
            return None
        
        trip_duration = len(self.gps_breadcrumbs)
        deviation_time = len(self.deviation_events)
        deviation_ratio = deviation_time / trip_duration if trip_duration > 0 else 0
        
        max_deviation = max([e['distance'] for e in self.deviation_events]) if self.deviation_events else 0
        
        return {
            'total_points': len(self.gps_breadcrumbs),
            'deviation_events': len(self.deviation_events),
            'deviation_ratio': deviation_ratio,
            'max_deviation': max_deviation,
            'was_anomalous': deviation_ratio > 0.3 or max_deviation > 1000
        }