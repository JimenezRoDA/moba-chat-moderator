import seaborn as sns
tab20b_hex_colors = sns.color_palette('tab20b', n_colors=20).as_hex()

PALETA_ATARDECER_GRIETA = {
    # Colores principales / de base 
    'azul_suave': tab20b_hex_colors[0],         
    'ocre_claro': tab20b_hex_colors[1],         
    'verde_oliva_suave': tab20b_hex_colors[2],  
    
    # Colores de acento 
    'naranja_ambar': tab20b_hex_colors[3],     
    'terracota': tab20b_hex_colors[4],        

    # Colores de advertencia / grave 
    'rojo_profundo': tab20b_hex_colors[13],
    'purpura_oscuro': tab20b_hex_colors[6],  

    # Colores adicionales para expandir 
    'crema_brillante': tab20b_hex_colors[7], 
    'azul_verdoso': tab20b_hex_colors[8],     
    'marrón_suave': tab20b_hex_colors[9],       

    # Colores de texto y fondo 
    'fondo_claro': '#F7F7F7',                   
    'texto_oscuro': '#333333'                    
}