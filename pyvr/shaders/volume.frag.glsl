#version 330 core

uniform sampler3D volume_texture;
uniform sampler3D normal_volume;
uniform sampler2D transfer_function_lut;  // Combined RGBA transfer function texture

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 camera_pos;
uniform float step_size;
uniform int max_steps;
uniform vec3 volume_min_bounds;
uniform vec3 volume_max_bounds;

uniform float ambient_light;
uniform float diffuse_light;
uniform vec3 light_position;
uniform vec3 light_target;

in vec2 uv;
out vec4 color;

// Coordinate conversion functions
vec3 world_to_texture_coords(vec3 world_pos) {
    return (world_pos - volume_min_bounds) / (volume_max_bounds - volume_min_bounds);
}

vec3 texture_to_world_coords(vec3 tex_pos) {
    return volume_min_bounds + tex_pos * (volume_max_bounds - volume_min_bounds);
}

vec2 screen_to_ndc_coords(vec2 screen_coord) {
    return screen_coord * 2.0 - 1.0;
}

vec2 ndc_to_screen_coords(vec2 ndc_coord) {
    return ndc_coord * 0.5 + 0.5;
}

bool is_valid_texture_coord(vec3 tex_coord) {
    return tex_coord.x >= 0.0 && tex_coord.x <= 1.0 &&
            tex_coord.y >= 0.0 && tex_coord.y <= 1.0 &&
            tex_coord.z >= 0.0 && tex_coord.z <= 1.0;
}

vec3 ray_direction(vec2 screen_coord) {
    vec2 ndc_coord = screen_to_ndc_coords(screen_coord);
    vec4 clip_coord = vec4(ndc_coord, -1.0, 1.0);
    vec4 eye_coord = inverse(projection_matrix) * clip_coord;
    eye_coord = vec4(eye_coord.xy, -1.0, 0.0);
    vec3 world_coord = (inverse(view_matrix) * eye_coord).xyz;
    return normalize(world_coord);
}

bool intersect_box(vec3 ray_origin, vec3 ray_dir, vec3 box_min, vec3 box_max, out float t_near, out float t_far) {
    vec3 inv_dir = 1.0 / ray_dir;
    vec3 t_min = (box_min - ray_origin) * inv_dir;
    vec3 t_max = (box_max - ray_origin) * inv_dir;
    
    vec3 t1 = min(t_min, t_max);
    vec3 t2 = max(t_min, t_max);
    
    t_near = max(max(t1.x, t1.y), t1.z);
    t_far = min(min(t2.x, t2.y), t2.z);
    
    return t_near <= t_far && t_far > 0.0;
}

void main() {
    vec3 ray_dir = ray_direction(uv);
    vec3 ray_origin = camera_pos;

    float t_near, t_far;
    if (!intersect_box(ray_origin, ray_dir, volume_min_bounds, volume_max_bounds, t_near, t_far)) {
        color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    t_near = max(t_near, 0.0);
    vec3 current_world_pos = ray_origin + ray_dir * t_near;
    float distance = t_far - t_near;

    vec4 accumulated_color = vec4(0.0);
    float accumulated_alpha = 0.0;

    for (int i = 0; i < max_steps && accumulated_alpha < 0.99; i++) {
        vec3 tex_coord = world_to_texture_coords(current_world_pos);
        // Swap x and z for texture sampling
        tex_coord = vec3(tex_coord.z, tex_coord.y, tex_coord.x);

        if (is_valid_texture_coord(tex_coord)) {
            float density = texture(volume_texture, tex_coord).r;
            
            // Single RGBA texture lookup for transfer function
            vec4 rgba = texture(transfer_function_lut, vec2(density, 0.5));
            vec3 rgb = rgba.rgb;
            float alpha = rgba.a;

            vec3 normal = texture(normal_volume, tex_coord).rgb;
            normal = normalize(normal);

            // Compute light direction from position to target (target is (0,0,0))
            vec3 light_dir = normalize(light_target - light_position);

            float diffuse_intensity = max(dot(normal, light_dir), 0.0);
            float light = ambient_light + diffuse_light * diffuse_intensity;

            vec4 sample_color = vec4(rgb * light, alpha);
            sample_color.rgb *= sample_color.a;
            accumulated_color += (1.0 - accumulated_alpha) * sample_color;
            accumulated_alpha += (1.0 - accumulated_alpha) * sample_color.a;
        }

        current_world_pos += ray_dir * step_size;
    }

    color = vec4(accumulated_color.rgb, accumulated_alpha);
}