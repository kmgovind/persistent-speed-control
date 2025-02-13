module Utilities

using Geodesy

# TODO: Complete methods

# Convert x,y position to coordinate
function pos2coord(pos_x, pos_y)
    return pos_x + pos_y
end

# Convert lat,long position to x,y position
function coord2pos(lat, lon)
end

"""
gps2coord(origin, current_pos)

Input:
- origin: [latitude, longitude] - axis system origin location
- current_pos: [latitude, longitude] - current position

Output:
- [xEast,yNorth] -  distances in km

Given the lat/long pair representing the origin of the domain, convert a provided lat/long pair to x/y coordinates relative to the origin.
"""
function gps2coord(origin, current_pos)
    origin_lla = LLA(origin[1], origin[2], 0.0);
    current_lla = LLA(current_pos[1], current_pos[2], 0.0);
    return ENU(current_lla, origin_lla, wgs84)[1:2]./1000 # return [E,N] as [x,y] in km
end
end # module Utilities