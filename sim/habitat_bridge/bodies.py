"""
Robot bodies for the Habitat home (shared by habitat_server.py and brain_client.py).

  spot   Habitat 3.0's Spot (the task's own robot): ~1.1 x 0.5 m, moves on a
         navmesh eroded by 0.25 m around three points along its body.
  rover  the Waveshare UGV Rover (the robot on order): 253 x 231 mm, 0.29 m
         high with its pan-tilt head. Spot stays in the scene only as a
         kinematic stand-in (its rendered camera and physics body are not
         used); the base moves on the rover's own navmesh (one circle of
         NAV_R around the centre, obstacles lower than HEIGHT_M and steps
         over MAX_CLIMB_M block it; it fits under tables and between chair
         legs), and contacts are measured with the rover's rectangle
         (habitat_server.Server._rover_contacts).

C. APPROXIMATIONS: the navmesh circle (0.13 m, about the half-length) lets
the rectangle's corners (0.17 m from the centre) clip into things while it
turns next to them; those count as contacts. The head camera stays at 0.35 m
(the person-distance estimate in robot/safety.py assumes that height).
"""
BODIES = {
    "spot": {"shape": "ellipse", "half_len": 0.55, "half_wid": 0.25,
             "antenna_ahead": 0.45, "eat_r": 0.6},
    "rover": {"shape": "rect", "half_len": 0.127, "half_wid": 0.116,
              "antenna_ahead": 0.15, "eat_r": 0.35,
              "nav_r": 0.13, "height": 0.30, "max_climb": 0.03},
}
